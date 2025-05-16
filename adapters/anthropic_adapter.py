from .adapter_base import AdapterBase
import anthropic
import os
from typing import List, Tuple, Dict, Callable
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, FunctionCall, FunctionResponse, Message, ThinkingResponse
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
from llm_platform.helpers.model_config import ModelConfig
import logging
import asyncio
import json
import inspect

class ClaudeStreamProcessor:
    """
    Processes Claude API stream events and accumulates them into:
    - thinking_text: The raw thinking text from thinking blocks
    - response_text: The actual response text shown to the user
    - tool_uses: A list of tools being used, with their parameters
    """
    def __init__(self):
        # Output fields
        self.thinking_text = ""
        self.thinking_responses: List[ThinkingResponse] = []
        self.response_text = ""
        self.tool_uses = []
        self.usage = {"model": "",
                      "completion_tokens": 0,
                      "prompt_tokens": 0 
                }
        self.stop_reason = None

        # Internal state tracking
        self._current_block_type = None
        self._current_block_index = None
        self._current_block_signature = None
        self._current_tool_name = None
        self._current_tool_id = None
        self._current_tool_json = ""

    def process_event(self, event):
        """Process a single event from the Claude stream."""
        if not hasattr(event, 'type'):
            return

        event_type = event.type

        if event_type == 'message_start':
            self._handle_message_start(event)
        elif event_type == 'content_block_start':
            self._handle_content_block_start(event)
        elif event_type == 'content_block_delta':
            self._handle_content_block_delta(event)
        elif event_type == 'content_block_stop':
            self._handle_content_block_stop(event)
        elif event_type == 'message_delta':
            self._handle_message_delta(event)

    def _handle_message_start(self, event):
        if not hasattr(event, 'message'):
            return
        
        self.usage["model"] = event.message.model
        self.usage["prompt_tokens"] = event.message.usage.input_tokens

    def _handle_content_block_start(self, event):
        """Handle the start of a content block."""
        if not hasattr(event, 'content_block'):
            return

        content_block = event.content_block

        # Set the current block type and index
        if hasattr(content_block, 'type'):
            self._current_block_type = content_block.type

        if hasattr(event, 'index'):
            self._current_block_index = event.index

        # For tool use blocks, store the tool information
        if self._current_block_type == 'tool_use':
            if hasattr(content_block, 'name'):
                self._current_tool_name = content_block.name

            if hasattr(content_block, 'id'):
                self._current_tool_id = content_block.id

            # Reset the JSON accumulator for the new tool
            self._current_tool_json = ""

    def _handle_content_block_delta(self, event):
        """Handle a delta update to a content block."""
        if not hasattr(event, 'delta'):
            return

        delta = event.delta

        if not hasattr(delta, 'type'):
            return

        delta_type = delta.type

        # Accumulate thinking text
        if delta_type == 'thinking_delta' and hasattr(delta, 'thinking'):
            self.thinking_text += delta.thinking

        # Accumulate response text
        elif delta_type == 'text_delta' and hasattr(delta, 'text'):
            self.response_text += delta.text

        # Accumulate tool JSON
        elif delta_type == 'input_json_delta' and hasattr(delta, 'partial_json'):
            self._current_tool_json += delta.partial_json

        # Save the signature of the current block
        elif delta_type == 'signature_delta' and hasattr(delta, 'signature'):
            self._current_block_signature = delta.signature

    def _handle_content_block_stop(self, event):
        """Handle the end of a content block."""
        # Process completed tool use blocks
        if self._current_block_type == 'tool_use' and self._current_tool_name:
            try:
                # Parse the complete JSON for the tool parameters
                parameters = json.loads(self._current_tool_json) if self._current_tool_json else {}

                # Add the tool use to our list
                tool_use = {
                    'name': self._current_tool_name,
                    'parameters': parameters
                }
                if self._current_tool_id:
                    tool_use['id'] = self._current_tool_id

                self.tool_uses.append(tool_use)
            except json.JSONDecodeError:
                # Handle malformed JSON by using empty parameters
                self.tool_uses.append({
                    'name': self._current_tool_name,
                    'id': self._current_tool_id,
                    'parameters': {}
                })

            # Reset the tool-specific state
            self._current_tool_name = None
            self._current_tool_id = None
            self._current_tool_json = ""

        elif self._current_block_type == 'thinking':
            self.thinking_responses.append(ThinkingResponse(content=self.thinking_text, id=self._current_block_signature))
            self.thinking_text = ""

        # Reset the block state
        self._current_block_type = None
        self._current_block_index = None
        self._current_block_signature = None

    def _handle_message_delta(self, event):
        self.usage["completion_tokens"] = event.usage.output_tokens
        self.stop_reason = event.delta.stop_reason

class AnthropicAdapter(AdapterBase):

    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)  
        self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        #self.async_client = anthropic.AsyncAnthropic(
        #    api_key=os.getenv("ANTHROPIC_API_KEY"),
        #)
        self.model_config = ModelConfig()

    def convert_conversation_history_to_adapter_format(self, 
                                                       the_conversation: Conversation, 
                                                       additional_parameters: Dict={}
                                                       ) -> List[Dict]:
        
        history = []

        # Add history of messages
        for message in the_conversation.messages:
            # Add message to history
            history_message = {"role": message.role, "content": message.content if message.content is not None else " "}
            
            # Add files
            if not message.files is None:

                # Ensure that history_message["content"] is a list, not a string
                if not isinstance(history_message["content"], list):
                    history_message["content"] = [{
                        "type": "text",
                        "text": history_message["content"]
                    }]

                for each_file in message.files:

                    # Images
                    if isinstance(each_file, ImageFile):

                        # Add the image to the content list
                        image_content = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{each_file.extension}",
                                "data": each_file.base64,
                            },
                        }

                        history_message["content"].append(image_content)
                    
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile)):

                        document_content = {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": each_file.text
                            },
                            "title": each_file.name,
                            "context": "This is a trustworthy document.",
                            "citations": {"enabled": additional_parameters.get("citations", False)}
                        }

                        history_message["content"].insert(0, document_content)

                    elif isinstance(each_file, PDFDocumentFile):

                        if (each_file.size < 32_000_000) and (each_file.number_of_pages < 100):

                            document_content = {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": each_file.base64
                                },
                                "title": each_file.name,
                                "context": "This is a trustworthy document.",
                                "citations": {"enabled": additional_parameters.get("citations", False)}
                            }

                        else:

                            document_content = {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": "text/plain",
                                    "data": each_file.text
                                },
                                "title": each_file.name,
                                "context": "This is a trustworthy document.",
                                "citations": {"enabled": additional_parameters.get("citations", False)}
                            }

                        history_message["content"].insert(0, document_content)
                    
                    else:
                        raise ValueError(f"Unsupported file type: {each_file.extension}")
                    
            # If there are any tool calls, add them into the content of the message
            if message.function_calls:
                # Ensure that history_message["content"] is a list, not a string
                if not isinstance(history_message["content"], list):
                    history_message["content"] = [{
                        "type": "text",
                        "text": history_message["content"]
                    }]
                
                # Add thinking to the history
                if message.thinking_responses:
                    for each_thinking_response in message.thinking_responses:
                        history_message["content"].insert(0, each_thinking_response.to_anthropic())

                # Add the tool calls to the history
                for each_function_call in message.function_calls:
                    history_message["content"].append(each_function_call.to_anthropic())

            history.append(history_message)

            # If there are function responses, add them into the history in a separate message -> https://docs.anthropic.com/en/docs/build-with-claude/tool-use
            if message.function_responses:
                history_message = {"role": "user", "content": []}
                for each_function_response in message.function_responses:
                    history_message["content"].append(each_function_response.to_anthropic())

                history.append(history_message)

        return history

    def request_llm(self, model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={}, 
                    **kwargs) -> Message:
        
        # Convert parameter `reasoning_efforts` to `thinking` parameter
        if 'reasoning' in kwargs:
            reasoning_effort = kwargs.pop('reasoning', {}).get('effort', 'low')
            """
                reasoning_effort 'low' -> thinking is not used at all
                reasoning_effort 'medium' -> thinking budget is 8000
                reasoning_effort 'high' -> thinking budget is 32000
            """
            if reasoning_effort in ['high', 'medium']:
                reasoning_effort_map = {'high': 32_000, 'medium': 8_000}
                budget_tokens = reasoning_effort_map[reasoning_effort]
                kwargs['thinking'] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens
                    }
                temperature = 1
                
        if model == 'claude-3-7-sonnet-20250219':
            kwargs['betas'] = ["output-128k-2025-02-19"]

        if functions is None:
            # Add history of messages
            history = self.convert_conversation_history_to_adapter_format(the_conversation, additional_parameters)

            # Check max_tokens and correct if necessary
            if 'max_tokens' in kwargs:
                kwargs['max_tokens'] = self.correct_max_tokens(model, history, kwargs['max_tokens'])

            tools = []
            # Web-search
            if additional_parameters.get("grounding", False):
                tools = [{
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 10,
                        }]

            stream = self.client.beta.messages.create(
                            model=model,
                            system = the_conversation.system_prompt,
                            messages=history,
                            temperature=temperature,
                            stream=True,
                            tools=tools,
                            **kwargs,
                        )
            
            processor = ClaudeStreamProcessor()
            for event in stream:
                processor.process_event(event)

        else:
            processor = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            temperature=temperature,
                            tool_output_callback=tool_output_callback,
                            **kwargs,
             )
            
        message = Message(role="assistant", content=processor.response_text, thinking_responses=processor.thinking_responses, usage=processor.usage)
        the_conversation.messages.append(message)
        
        return message


    def count_tokens(self, model, messages, tools=None):
        """Count tokens for a given message list."""
        if tools:
            result = self.client.messages.count_tokens(
                model=model,
                messages=messages,
                tools=tools
            )
        else:
            result = self.client.messages.count_tokens(
                model=model,
                messages=messages
            )
        return result.input_tokens
    
    def correct_max_tokens(self, model, messages, max_tokens, tools=None):
        """Check max_tokens and correct if necessary
        """
        request_tokens = self.count_tokens(model, messages, tools=tools)
        context_window_size = self.model_config[model].context_window
        if request_tokens + max_tokens >= context_window_size:
            updated_max_tokens = context_window_size - request_tokens - 1000      # 1000 tokens for the answer
            logging.warning(f"Request tokens ({request_tokens}) + Max tokens ({max_tokens}) exceeds context window ({context_window_size}). Correcting max tokens to {updated_max_tokens}.")
            return updated_max_tokens
        else:
            return max_tokens

    def voice_to_text(self, audio_file):
        raise NotImplementedError("Anthropic does not support voice to text")

    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        raise NotImplementedError("Anthropic does not support image generation")

    def _convert_func_to_tool(self, func: Callable) -> Dict:
        # Get function signature
        sig = inspect.signature(func)

        # Create parameters dictionary
        parameters = {}
        required_params = []

        # Analyze each parameter
        for param_name, param in sig.parameters.items():
            param_info = {}

            # Get parameter type annotation if available
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_info['type'] = 'string'
                elif param.annotation == int:
                    param_info['type'] = 'integer'
                elif param.annotation == float:
                    param_info['type'] = 'number'
                elif param.annotation == bool:
                    param_info['type'] = 'boolean'
                elif param.annotation == list:
                    param_info['type'] = 'array'
                elif param.annotation == dict:
                    param_info['type'] = 'object'
                else:
                    param_info['type'] = 'string'  # default to string for unknown types
            else:
                param_info['type'] = 'string'  # default to string if no type annotation

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)

            parameters[param_name] = param_info

        # Create the tool dictionary
        tool = {
                'name': func.__name__,
                'description': func.__doc__ or '',
                'input_schema': {
                    'type': 'object',
                    'properties': parameters,
                    'required': required_params,
                },
        }
        return tool

    def _convert_function_to_tool(self, func: BaseTool) -> Dict:
        if isinstance(func, BaseTool):
            # Handle the case where func is a BaseTool
            tool = func.to_params(provider='anthropic')
        elif callable(func):
            tool = self._convert_func_to_tool(func)
        else:
            raise TypeError("func must be either a BaseTool or a function")
        return tool
    

    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool], 
                                   temperature: int=0,  
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs):
        
        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        messages = self.convert_conversation_history_to_adapter_format(the_conversation)

        # Web-search
        if additional_parameters.get("grounding", False):
            tools.append({
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 10,
                    })

        stream = self.client.beta.messages.create(
                            messages=messages,
                            model=model,
                            system = the_conversation.system_prompt,
                            temperature=temperature,
                            tools=tools,
                            stream=True,
                            **kwargs,
                        )
        
        processor = ClaudeStreamProcessor()
        for event in stream:
            processor.process_event(event)

        # If there are no tool calls, we can return the response
        if processor.stop_reason != "tool_use":
            return processor
        
        function_call_records = []
        function_response_records = []

        # Extract all requests to use the tools from the response
        for each_tool in processor.tool_uses:
            # Find the requested function
            function_index = next((i for i, tool in enumerate(tools) if tool['name'] == each_tool["name"]), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_tool['name']} not found in tools")
            function = functions[function_index]

            # Get all function parameters
            function_parameters = []
            for key, value in tools[function_index].get('input_schema', {}).get('properties', {}).items():
                function_parameters.append(each_tool["parameters"].get(key, ""))

            # Call the function
            function_response = function(*function_parameters)

            # Save function_call and function_response records
            function_call_record = FunctionCall(id=each_tool['id'],
                                                name=each_tool['name'],
                                                arguments=each_tool['parameters'],
                                                )
            function_call_records.append(function_call_record)

            function_response_record = FunctionResponse(name=each_tool['name'],
                                                        id=each_tool['id'],
                                                        response=function_response
                                                        )
            function_response_records.append(function_response_record)

            if tool_output_callback:
                tool_output_callback(each_tool['name'],
                                     function_parameters,
                                     function_response
                                    )

        message = Message(role="assistant", 
                                    content=processor.response_text,
                                    thinking_responses=processor.thinking_responses,
                                    function_calls=function_call_records,
                                    function_responses=function_response_records,
                                    usage=processor.usage
                                )
        the_conversation.messages.append(message)

        final_response = self.request_llm_with_functions(model, the_conversation, functions, temperature, tool_output_callback=tool_output_callback, **kwargs)
        return final_response


    def request_llm_computer_use(self, model: str, 
                                 the_conversation: Conversation, 
                                 functions: List[Callable], 
                                 temperature: int=0,  
                                 **kwargs):
        
        if not functions is None:
            tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        
        tools += [
            #{
            #    "type": "computer_20241022",
            #    "name": "computer",
            #    "display_width_px": 3840,
            #    "display_height_px": 2160,
            #    "display_number": 1,
            #},
            {
                "type": "text_editor_20241022",
                "name": "str_replace_editor"
            },
            {
                "type": "bash_20241022",
                "name": "bash"
            },
        ]
        
        
        
        messages = self.convert_conversation_history_to_adapter_format(the_conversation)
        system_prompt = the_conversation.system_prompt

        chat_response = self.client.beta.messages.create(
                        model=model,
                        messages=messages,
                        #system=system_prompt,
                        tools=tools,
                        #temperature=temperature,
                        betas=["computer-use-2024-10-22"],
                        **kwargs,
                    )
        
        usage = {"model": model,
                "completion_tokens": chat_response.usage.output_tokens,
                "prompt_tokens": chat_response.usage.input_tokens}
        
        # If there are no tool calls, we can return the response
        if chat_response.stop_reason != "tool_use":
            return chat_response
        
        # Extract all requests to use the tools from the response
        tool_use_requests = [each for each in chat_response.content if getattr(each, 'type', '') == 'tool_use']
        
        #Save request to use tool by assistant to the history of messages
        message = Message(role="assistant", 
                            content=chat_response.content, 
                            usage=usage)
        the_conversation.messages.append(message)
        
        for each_tool in tool_use_requests:
            # Find the requested function
            function_index = next((i for i, tool in enumerate(tools) if tool['name'] == each_tool.name), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_tool.name} not found in tools")
            function = functions[function_index]

            # Get all function parameters
            """
            BetaToolUseBlock(
                id="toolu_01MeD8RaAVZ4E1Qhq5YvgQkA",
                input={"action": "mouse_move", "coordinate": [400, 386]},
                name="computer",
                type="tool_use",
            ),
            """
            function_parameters = []
            if each_tool.name == "computer":
                for key, value in each_tool.input.items():
                    function_parameters.append(value)
            else:
                for key, value in tools[function_index].get('input_schema', {}).get('properties', {}).items():
                    function_parameters.append(getattr(each_tool, "input", {}).get(key, ""))

            # Call the function
            function_response = function(*function_parameters)

            message = Message(role="user", 
                                content=[{
                                            "type": "tool_result",
                                            "tool_use_id": getattr(each_tool, "id", ""),
                                            "content": str(function_response)
                                        }],
                                usage=usage)
            the_conversation.messages.append(message)

        final_response = self.request_llm_with_functions(model, the_conversation, functions, temperature, **kwargs)
        return final_response


    def get_models(self) -> List[str]:
        # Retrieve the list of models
        response = self.client.models.list_models()

        return [model['id'] for model in response['data']]
