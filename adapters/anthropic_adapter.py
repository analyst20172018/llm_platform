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
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")
                    
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
            
            stream = self.client.beta.messages.create(
                            model=model,
                            system = the_conversation.system_prompt,
                            messages=history,
                            temperature=temperature,
                            stream=True,
                            **kwargs,
                        )
            
            usage = {"model": model,
                    "completion_tokens": 0, #response.usage.output_tokens,
                    "prompt_tokens": 0 #response.usage.input_tokens
                }
            
            # Response from the stream
            thinking_text = ""
            response_text = ""
            for event in stream:
                # Get content (thinking or text)
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        thinking_text += event.delta.thinking

                    if event.delta.type == "text_delta":
                        response_text += event.delta.text
                    
                # Get usage
                if getattr(event, 'message', None):
                    if getattr(event.message, 'usage', None):
                        usage['prompt_tokens'] += getattr(event.message.usage, 'input_tokens', 0)
                        usage['completion_tokens'] += getattr(event.message.usage, 'output_tokens', 0)
                if getattr(event, 'usage', None):
                    usage['prompt_tokens'] += getattr(event.usage, 'input_tokens', 0)
                    usage['completion_tokens'] += getattr(event.usage, 'output_tokens', 0)

            full_response = response_text

            if thinking_text:
                thinking_responses = [ThinkingResponse(content=thinking_text)]
            else:
                thinking_responses = []
            message = Message(role="assistant", content=full_response, thinking_responses=thinking_responses, usage=usage)

        else:
            response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            temperature=temperature,
                            tool_output_callback=tool_output_callback,
                            **kwargs,
             )
            
            # Usually response is in response.content[0].text, but here Anthropic may answer with multiple messages
            full_response = ''.join([each.text for each in response.content])
            thinking_text = ""
        
            usage = {"model": model,
                    "completion_tokens": response.usage.output_tokens,
                    "prompt_tokens": response.usage.input_tokens
                    }
        
            message = Message(role="assistant", content=full_response, usage=usage, thinking_responses=[])
        
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

        # Check max_tokens and correct if necessary
        #if 'max_tokens' in kwargs:
        #    kwargs['max_tokens'] = self.correct_max_tokens(model, messages, kwargs['max_tokens'], tools=tools)

        chat_response = self.client.beta.messages.create(
                        model=model,
                        messages=messages,
                        system = the_conversation.system_prompt,
                        tools=tools,
                        temperature=temperature,
                        **kwargs,
                    )
        
        usage = {"model": model,
                    "completion_tokens": chat_response.usage.output_tokens,
                    "prompt_tokens": chat_response.usage.input_tokens}
        
        # If there are no tool calls, we can return the response
        if chat_response.stop_reason != "tool_use":
            return chat_response
        
        function_call_records = []
        function_response_records = []

        # Extract the assistant's text response
        if isinstance(chat_response.content, str):
            text_assistant_answer = chat_response.content
        elif isinstance(chat_response.content, list):
            text_answers = [each.text for each in chat_response.content if getattr(each, 'type', '') == 'text']
            text_assistant_answer = '\n'.join(text_answers)
        else:
            raise ValueError(f"Unsupported response type: {type(chat_response.content)}. Must be str or list")
        
        # Extract all requests to use the tools from the response
        tool_use_requests = [each for each in chat_response.content if getattr(each, 'type', '') == 'tool_use']
        for each_tool in tool_use_requests:
            # Find the requested function
            function_index = next((i for i, tool in enumerate(tools) if tool['name'] == each_tool.name), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_tool.name} not found in tools")
            function = functions[function_index]

            # Get all function parameters
            function_parameters = []
            for key, value in tools[function_index].get('input_schema', {}).get('properties', {}).items():
                function_parameters.append(getattr(each_tool, "input", {}).get(key, ""))

            # Call the function
            function_response = function(*function_parameters)

            # Save function_call and function_response records
            function_call_record = FunctionCall(id=getattr(each_tool, "id", ""),
                                                name=each_tool.name,
                                                arguments=json.dumps(getattr(each_tool, "input", {}))
                                                )
            function_call_records.append(function_call_record)

            function_response_record = FunctionResponse(name=each_tool.name,
                                                        id=getattr(each_tool, "id", ""),
                                                        response=function_response
                                                        )
            function_response_records.append(function_response_record)

            if tool_output_callback:
                tool_output_callback(each_tool.name,
                                     function_parameters,
                                     function_response
                                    )
                
        # Extract all responses with the thinking
        thinking_responses_history = []
        thinking_responses_from_llm = [each for each in chat_response.content if getattr(each, 'type', '') in ['thinking', 'redacted_thinking']]
        for each_thinking_response in thinking_responses_from_llm:
            new_thinking_response = ThinkingResponse(content=each_thinking_response.thinking, id=each_thinking_response.signature)
            thinking_responses_history.append(new_thinking_response)


        message = Message(role="assistant", 
                                    content=text_assistant_answer,
                                    thinking_responses=thinking_responses_history,
                                    function_calls=function_call_records,
                                    function_responses=function_response_records,
                                    usage=usage
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

        print(tools)
        print(messages)

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
        
        print(chat_response)
        
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
            print(f"Function parameters: {function_parameters}")
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
