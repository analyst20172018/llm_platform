from .adapter_base import AdapterBase
from openai import OpenAI
import requests
import os
from typing import List, Tuple, Callable, Dict
import logging
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse, ThinkingResponse
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
import json
import inspect

class GrokAdapter(AdapterBase):
    
    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)   
        self.client = OpenAI(
            api_key = os.getenv("XAI_API_KEY"),
            base_url = "https://api.x.ai/v1",
        )

        self.completions_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
        }
        
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
                "content": [{
                        "type": "text",
                        "text": message.content
                }]
            }

            # If there is an attribute tool_calls in message, then add it to history. 
            if message.function_calls:
                function_calls_message = {
                    'content': '',
                    'refusal': None,
                    'role': 'assistant',
                    'tool_calls': [],
                }
                
                for each_function_call in message.function_calls:
                    function_calls_message["tool_calls"].append(each_function_call.to_grok())
                
                history.append(function_calls_message)

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
                    elif isinstance(each_file, AudioFile):
                        raise NotImplementedError("Grok does not support audio files")

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
                    history.append(each_response.to_grok())

        return history, kwargs

    def _create_parameters_for_calling_llm(self, 
                        model: str,
                        the_conversation: Conversation,
                        additional_parameters: Dict={},
                        **kwargs
                        ) -> Dict:
        messages, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model, **kwargs)

        parameters = {
            "model": model,
            "messages": messages,
            "tools": [],
        }

        # Add `reasoning` parameter if exists
        if 'reasoning' in kwargs:
            parameters['reasoning_effort'] = kwargs.pop('reasoning', {}).get('effort', 'low')

        # Add web search parameter if exists
        if additional_parameters.get("web_search", False):
            parameters['search_parameters'] = {"mode": "auto"}

        return parameters

    def request_llm(self, model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    temperature: int=0, 
                    tool_output_callback: Callable=None, 
                    additional_parameters: Dict={},
                    **kwargs) -> Message:
        """Requests a completion from the Language Model.
                    This method sends the current conversation to the LLM and retrieves a response.
                    It can handle both standard chat completions and completions involving function calls (tools).
                    The conversation history is updated with the LLM's response.
                    Args:
                        model (str): The identifier of the LLM model to be used.
                        the_conversation (Conversation): The conversation object containing the history
                            of messages. This object will be updated with the LLM's response.
                        functions (List[BaseTool], optional): A list of tools (functions) that the LLM
                            can choose to call. Defaults to None, indicating no functions are available.
                        temperature (int, optional): Controls the randomness of the LLM's output.
                            default: 0; min: 0; max: 2 What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                        tool_output_callback (Callable, optional): A callback function that is invoked
                            if the LLM decides to call a tool. This function is responsible for
                            executing the tool and returning its output. Required if `functions`
                            are provided. Defaults to None.
                        additional_parameters (Dict, optional): A dictionary of additional parameters
                            to be passed to the LLM API. Defaults to an empty dictionary.
                        **kwargs: Arbitrary keyword arguments that will be passed to the underlying
                            LLM API calls.
                    Returns:
                        Message: A Message object representing the LLM's response, including its
                            content and usage statistics. This message is also appended to
                            `the_conversation.messages`.
        """
        if functions is None:
            parameters = self._create_parameters_for_calling_llm(
                            model, 
                            the_conversation, 
                            additional_parameters,
                            **kwargs
                        )

            response = requests.post(self.completions_url, headers=self.headers, json=parameters)
            response = response.json()
        else:
            response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            **kwargs,
            )
        
        usage = {"model": model,
                "completion_tokens": response['usage']['completion_tokens'],
                "prompt_tokens": response['usage']['total_tokens']}
        
        thinking_responses = []
        if 'reasoning_content' in response['choices'][0]['message']:
            thinking_responses.append(ThinkingResponse(content=response['choices'][0]['message']['reasoning_content'], id=response['id']))
        
        message = Message(
            role="assistant", 
            content=response['choices'][0]['message']['content'], 
            thinking_responses=thinking_responses,
            usage=usage
        )
        the_conversation.messages.append(message)
        
        return message

    def voice_to_text(self, audio_file):
        raise NotImplementedError("OpenRoute does not support voice to text")

    def generate_image(self, prompt: str, n: int=1, **kwargs) -> List[ImageFile]:
        response = self.client.images.generate(
                model="grok-2-image",
                prompt=prompt,
                n=n,
                response_format="b64_json",
            )

        output_images = [ImageFile.from_base64(base64_str=image_data.b64_json, file_name="image.png") for image_data in response.data]

        return output_images

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
                                   additional_parameters: Dict={},
                                   **kwargs):
        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        parameters = self._create_parameters_for_calling_llm(
                            model, 
                            the_conversation, 
                            additional_parameters,
                            **kwargs
                        )
        
        parameters['tools'] += tools
        
        response = requests.post(self.completions_url, headers=self.headers, json=parameters)
        response = response.json()

        usage = {"model": model,
                "completion_tokens": response['usage']['completion_tokens'],
                "prompt_tokens": response['usage']['total_tokens']}
        
        assistant_message = response['choices'][0]['message']

        # Save tool_calls parameter from the openai answer for the history
        if assistant_message.get('tool_calls', None) is not None:
            function_call_records = [FunctionCall.from_grok(each_tool_call) for each_tool_call in assistant_message['tool_calls']]
        else: 
            function_call_records = []

        # If there are no tool calls, we can return the response
        if assistant_message.get('tool_calls', None) is None:
            return response

        function_response_records = []
        for each_tools_call in assistant_message.get('tool_calls', []):

            tool_call_id = each_tools_call['id']
            tool_function_name = each_tools_call['function']['name']
            tool_arguments = json.loads(each_tools_call['function']['arguments'])

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

        thinking_responses = []
        if 'reasoning_content' in response['choices'][0]['message']:
            thinking_responses.append(ThinkingResponse(content=response['choices'][0]['message']['reasoning_content'], id=response['id']))
        
        message = Message(
            role="assistant", 
            content=response['choices'][0]['message']['content'], 
            thinking_responses=thinking_responses,
            function_calls=function_call_records,
            function_responses=function_response_records,
            usage=usage
        )
        the_conversation.messages.append(message)

        final_response = self.request_llm_with_functions(model, the_conversation, functions, tool_output_callback=tool_output_callback, **kwargs)
        
        return final_response

    def get_models(self) -> List[str]:
        raise NotImplementedError("Not implemented yet")
