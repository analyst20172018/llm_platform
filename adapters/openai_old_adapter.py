from .adapter_base import AdapterBase
from openai import OpenAI
import os
import json
from typing import List, Tuple, Callable, Dict
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile)
import inspect

class OpenAIOldAdapter(AdapterBase):
    
    def __init__(self):
        super().__init__()   
        self.client = OpenAI()
        

    def convert_conversation_history_to_adapter_format(self, 
                        the_conversation: Conversation, 
                        model: str, 
                        **kwargs):
        
        # Add system prompt as the message from the user "system"
        history = [{"role": "system", "content": the_conversation.system_prompt}]

        # Add history of messages
        for message in the_conversation.messages:

            history_message = {
                "role": message.role, 
                "content": []
            }

            if message.content and message.content != "":
                history_message["content"].append({
                    "type": "text",
                    "text": message.content
                })

            # If there is an attribute tool_calls in message, then add it to history. 
            if message.function_calls:
                tool_calls = []
                for each_function_call in message.function_calls:
                    tool_calls.append(each_function_call.to_openai_old())
                history.append({"role": "assistant", "tool_calls": tool_calls})

            # Add all function responses to the history
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(each_response.to_openai_old())

            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Add the image to the content list
                        image_content = {"type": "image_url", 
                                         "image_url": {"url": f"data:image/{each_file.extension};base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    if isinstance(each_file, AudioFile):

                        # Audio URLs are only allowed for messages with role 'user', message with role 'assistant' may not contain an audio URL.
                        if message.role == "assistant":
                            continue

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
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
                        
                        # Add the text document to the history as a text in XML tags
                        new_text_content = {
                            "type": "text",
                            "text": f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        }
                        history_message["content"].insert(0, new_text_content)

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            if len(history_message["content"]) > 0:
                history.append(history_message)

        return history, kwargs


    def define_parameters_for_chat_completion_request(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={},
                    **kwargs) -> Dict:
        history, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model, **kwargs)

        chat_completion_parameters = {
            "model": model,
            "messages": history,
            "temperature": temperature,
            "audio": {"voice": "alloy", "format": "wav"},
            "modalities": ["text", "audio"],
        }
        chat_completion_parameters.update(kwargs)

        if "response_modalities" in additional_parameters:
                chat_completion_parameters["modalities"] = additional_parameters["response_modalities"]

        return chat_completion_parameters


    def request_llm(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=[], 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={},
                    **kwargs) -> Message:

        if functions is None or len(functions) == 0:
            chat_completion_parameters = self.define_parameters_for_chat_completion_request(
                model=model,
                the_conversation=the_conversation,
                temperature=temperature,
                additional_parameters=additional_parameters,
                **kwargs
            )

            response = self.client.chat.completions.create(**chat_completion_parameters)

        else:
            response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            additional_parameters=additional_parameters,
                            **kwargs,
             )
            
        usage = {"model": model,
                 "completion_tokens": response.usage.completion_tokens,
                 "prompt_tokens": response.usage.prompt_tokens}
        
        response_text = response.choices[0].message.content or ""

        files = []
        # Audio response
        if response.choices[0].message.audio:
            audio_response_file = AudioFile.from_base64(base64_str=response.choices[0].message.audio.data,
                                                        file_name="response.wav")
            files.append(audio_response_file)

            # Add transcript
            if response.choices[0].message.audio.transcript:
                response_text += f"\n{response.choices[0].message.audio.transcript}"
        
        message = Message(role="assistant", content=response_text, files=files, usage=usage)
        the_conversation.messages.append(message)
        
        return message


    def voice_to_text(self, audio_file):
        raise NotImplementedError("OpenRoute does not support voice to text")


    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        NotImplementedError("Not implemented yet")


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
                                   functions: List[BaseTool], 
                                   temperature: int=0,  
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs):
        
        chat_completion_parameters = self.define_parameters_for_chat_completion_request(
                model=model,
                the_conversation=the_conversation,
                temperature=temperature,
                additional_parameters=additional_parameters,
                **kwargs
            )
        
        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        chat_completion_parameters["tools"] = tools

        chat_response = self.client.chat.completions.create(**chat_completion_parameters)

        usage = {"model": model,
                "completion_tokens": chat_response.usage.completion_tokens,
                "prompt_tokens": chat_response.usage.prompt_tokens}
        
        assistant_message = chat_response.choices[0].message

        # Save tool_calls parameter from the openai answer for the history
        if getattr(assistant_message, 'tool_calls', None) is not None:
            function_call_records = [FunctionCall.from_openai_old(each_tool_call) for each_tool_call in assistant_message.tool_calls]
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
                                                        response=str(function_response))
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

        final_response = self.request_llm_with_functions(model, 
                                                         the_conversation, 
                                                         functions, 
                                                         tool_output_callback=tool_output_callback, 
                                                         additional_parameters=additional_parameters, 
                                                         **kwargs)
        
        return final_response


    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")

    