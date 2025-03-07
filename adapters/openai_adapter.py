from .adapter_base import AdapterBase
from openai import OpenAI, AsyncOpenAI
import json
from typing import List, Tuple, Dict, Callable
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, FunctionCall, FunctionResponse, Message
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
import logging
import inspect
import asyncio

class OpenAIAdapter(AdapterBase):
    
    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)  
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        
    def convert_conversation_history_to_adapter_format(self, 
                        the_conversation: Conversation, 
                        model: str, 
                        **kwargs):
        
        # Add system prompt as the message from the user "system" for OpenAI (for o1 models as "developer")
        if model in ['o1', 'o3-mini']: # 'o1-mini'
            history = [{"role": "developer", "content": the_conversation.system_prompt}]
        elif model == 'o1-mini':
            logging.warning("The model 'o1-mini' does not support system prompt")
            history = []
        else:
            history = [{"role": "system", "content": the_conversation.system_prompt}]

        # Add history of messages
        for message in the_conversation.messages:

            history_message = {"role": message.role, "content": message.content}

            # If there is an attribute tool_calls in message, then add it to history. 
            if message.function_calls:
                history_message["tool_calls"] = [each_call.to_openai() for each_call in message.function_calls]

            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Check if the model supports images
                        if model == 'o1-mini':
                            logging.warning("The model 'o1-mini' does not support images yet. Image is ignored.")
                            continue

                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{
                                "type": "text",
                                "text": history_message["content"]
                            }]

                        # Add the image to the content list
                        image_content = {"type": "image_url", 
                                         "image_url": {"url": f"data:image/{each_file.extension};base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    if isinstance(each_file, AudioFile):
                        # Update kwargs as needed
                        assert "gpt-4o-audio" in model
                        if 'modalities' not in kwargs:
                            kwargs['modalities'] = ["text"] #["text", "audio"]
                        if 'audio' not in kwargs:
                            kwargs['audio'] = {"voice": "alloy", "format": "wav"}

                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]

                        # Add audio to history
                        audio_content = {
                            "type": "input_audio",
                            "input_audio": {
                                "data": each_file.base64,
                                "format": "mp3"
                            }
                        }
                        history_message["content"].append(audio_content)
                    
                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile)):
                        
                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{
                                "type": "text",
                                "text": history_message["content"]
                            }]

                        # Add the text document to the history as a text in XML tags
                        new_text_content = {
                            "type": "text",
                            "text": f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        }
                        history_message["content"].insert(0, new_text_content)

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            history.append(history_message)

            # Add all function responses to the history
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(each_response.to_openai())

        return history, kwargs

    def request_llm(self, model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    temperature: int=0, 
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={},
                    **kwargs):
        
        if additional_parameters:
            logging.warning("Additional parameters is not supported by OpenAI API")

        # Remove 'max_tokens' and 'temperature' from kwargs if 'o1-' is in the model name
        if model.lower() in ['o1', 'o3-mini']:
            kwargs.pop('max_tokens', None)
            kwargs.pop('temperature', None)
        else:
            # Add temperature to kwargs
            kwargs['temperature'] = temperature


        if functions is None:
            messages, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model, **kwargs)
            response = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            **kwargs,
                            )
        else:
             response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            **kwargs,
             )

        usage = {"model": model,
                 "completion_tokens": response.usage.completion_tokens,
                 "prompt_tokens": response.usage.prompt_tokens}
        
        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        
        return response.choices[0].message.content

    def generate_image(self, prompt: str, n=1, **kwargs):
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=kwargs.get('size', '1024x1024'), # ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
                quality=kwargs.get('quality', 'hd'), #["standard", "hd"]
                n=n,
            )

            #image_url = response.data[0].url
            return response

    def voice_to_text(self, audio_file, response_format="text", language="en"):
        assert response_format in ["text", "srt", "verbose_json"]
        self.latest_usage = None

        transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format=response_format,
                language=language
                )
        return transcript

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
            'function': {
                'name': func.__name__,
                'description': func.__doc__ or '',
                'parameters': {
                    'type': 'object',
                    'properties': parameters,
                    'required': required_params,
                    "additionalProperties": False
                },
                "strict": True
            },
            'type': 'function',
        }
        return tool

    def _convert_function_to_tool(self, func: BaseTool | Callable) -> Dict:
        # Convert the function to a tool for OpenAI
        if isinstance(func, BaseTool):
            # Handle the case where func is a BaseTool
            tool = {
                'function': func.to_params(provider='openai'),
                'type': 'function',
            }
        
        elif callable(func):
            tool = self._convert_func_to_tool(func)
        else:
            raise TypeError("func must be either a BaseTool or a function")
        return tool

    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool | Callable], 
                                   tool_output_callback: Callable=None,
                                   **kwargs):
        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        messages, _ = self.convert_conversation_history_to_adapter_format(the_conversation, model)

        chat_response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        **kwargs,
                        )
        
        usage = {"model": model,
                "completion_tokens": chat_response.usage.completion_tokens,
                "prompt_tokens": chat_response.usage.prompt_tokens}
        
        assistant_message = chat_response.choices[0].message

        # Save tool_calls parameter from the openai answer for the history
        if getattr(assistant_message, 'tool_calls', None) is not None:
            function_call_records = [FunctionCall.from_openai(each_tool_call) for each_tool_call in assistant_message.tool_calls]
        else: 
            function_call_records = []

        # If there are no tool calls, we can return the response
        if getattr(assistant_message, 'tool_calls', None) is None:
            return chat_response

        function_response_records = []
        for each_tools_call in getattr(assistant_message, 'tool_calls', []):

            tool_call_id = each_tools_call.id
            tool_function_name = each_tools_call.function.name
            tool_arguments = json.loads(each_tools_call.function.arguments)

            # Find the requested function
            function_index = next((i for i, tool in enumerate(tools) if tool['function']['name'] == tool_function_name), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_tools_call.function.name} not found in tools")
            function = functions[function_index]

            # Get all function parameters
            function_parameters = []
            for key, value in tools[function_index]['function'].get('parameters', {}).get('properties', {}).items():
                function_parameters.append(tool_arguments[key])

            # Call the function
            function_response = function(*function_parameters)

            function_response_record = FunctionResponse(name=tool_function_name,
                                                        id=tool_call_id,
                                                        response=function_response)
            function_response_records.append(function_response_record)
            
            if tool_output_callback:
                tool_output_callback(tool_function_name,
                                     function_parameters,
                                     function_response
                                    )

        message = Message(role=assistant_message.role, 
                            content=assistant_message.content,
                            function_calls=function_call_records,
                            function_responses=function_response_records,
                            usage=usage
                            )
        the_conversation.messages.append(message)

        final_response = self.request_llm_with_functions(model, the_conversation, functions, tool_output_callback=tool_output_callback, **kwargs)

        return final_response

    def get_models(self) -> List[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]
