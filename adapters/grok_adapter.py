from .adapter_base import AdapterBase
from xai_sdk import Client
from xai_sdk.chat import user, system, image, assistant, tool, tool_result
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters
from xai_sdk.tools import web_search, x_search, code_execution
import os
from typing import Any, Callable, Dict, List, Tuple
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse, ThinkingResponse
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile)
import json
import inspect
from loguru import logger

class GrokAdapter(AdapterBase):
    
    ROLE_MAPPLING = {
        "user": chat_pb2.MessageRole.ROLE_USER,
        "assistant": chat_pb2.MessageRole.ROLE_ASSISTANT,
    }

    def __init__(self):
        super().__init__()   
        self.client = Client(api_key = os.getenv("XAI_API_KEY"))

    def convert_conversation_history_to_adapter_format(self,
                        chat,
                        the_conversation: Conversation, 
                        model: str, 
                        **kwargs):
        
        # Add system prompt as the message from the user "system"
        chat.append(system(the_conversation.system_prompt))

        # Add history of messages
        for message in the_conversation.messages:

            text_content = chat_pb2.Content(text=message.content)
            message_parameters = {
                "role": self.ROLE_MAPPLING.get(message.role),
                "content": [text_content],
            }

            # If there is an attribute tool_calls in message, then add it to history. 
            if message.function_calls:
                tool_calls = []
                for each_function_call in message.function_calls:
                    tool_call = chat_pb2.ToolCall(
                        id=each_function_call.id,
                        function=chat_pb2.FunctionCall(
                            name=each_function_call.name,
                            arguments=each_function_call.arguments
                        ),
                    )
                    tool_calls.append(tool_call)

                message_parameters["tool_calls"] = tool_calls
                
            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        image_parameter = image(image_url=f"data:image/{each_file.extension};base64,{each_file.base64}", detail="high")
                        message_parameters["content"].append(image_parameter)
                    
                    # Audio
                    elif isinstance(each_file, AudioFile):
                        raise NotImplementedError("Grok does not support audio files")

                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
                        
                        # Add the text document to the history as a text in XML tags
                        document_content_as_text = f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        document_content = chat_pb2.Content(text=document_content_as_text)

                        # Add document content to the arguments into message content
                        message_parameters["content"].append(document_content)

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            chat.append(
                chat_pb2.Message(**message_parameters)
            )

            # Add all function responses to the chat
            if message.function_responses:
                for each_response in message.function_responses:
                    chat.append(tool_result(result = str(each_response.response)))

        return chat, kwargs

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        additional_parameters: Dict = None,
    ) -> Dict:
        """
        Constructs the dictionary of parameters for an OpenAI API call.

        Args:
            model: The model identifier.
            additional_parameters: A dictionary of extra parameters like 'web_search'.
            use_previous_response_id: Flag to enable delta-based conversation updates.
            **kwargs: Additional keyword arguments to pass to the API.

        Returns:
            A dictionary of parameters ready for the OpenAI client.
        """
        #model_object = self.model_config[model]
        additional_parameters = additional_parameters or {}

        parameters = {
            "model": model,
            "tools": [],
        }

        reserved = {
            "web_search",
            "code_execution",
            "response_modalities",
            "citations_enabled",
            "url_context",
            "structured_output",
            "reasoning",
            "text",
        }
        for key, value in additional_parameters.items():
            if key in reserved:
                continue
            parameters[key] = value

        # Add web search parameter if exists
        if additional_parameters.get("web_search", False):
            parameters['tools'] += [
                web_search(),
                x_search(),
            ]

        # Add code execution parameter if exists
        if additional_parameters.get("code_execution"):
            parameters['tools'].append(
                code_execution(),
            )

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
        if additional_parameters is None:
            additional_parameters = {}

        if temperature not in (None, 0) and "temperature" not in additional_parameters:
            additional_parameters["temperature"] = temperature

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)

        if functions is None:
            
            parameters = self._create_parameters_for_calling_llm(
                model=model,
                additional_parameters=additional_parameters,
            )

            # Create chat with the arguments
            chat = self.client.chat.create(**parameters)

            chat, _ = self.convert_conversation_history_to_adapter_format(chat, the_conversation, model)

            response = chat.sample()
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
        
        thinking_responses = []
        if response.reasoning_content:
            thinking_responses.append(ThinkingResponse(content=response.reasoning_content, id=response.id))
        
        message = Message(
            role="assistant", 
            content=response.content, 
            thinking_responses=thinking_responses,
            usage=usage
        )
        the_conversation.messages.append(message)
        
        return message

    def request_llm_with_functions(self, 
                                    model: str, 
                                    the_conversation: Conversation, 
                                    functions: List[BaseTool | Callable], 
                                    tool_output_callback: Callable=None,
                                    additional_parameters: Dict={},
                                    **kwargs):
        if additional_parameters is None:
            additional_parameters = {}

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)
        tool_definitions = [self._convert_function_to_tool(each_function) for each_function in functions]

        parameters = self._create_parameters_for_calling_llm(
                model=model,
                additional_parameters=additional_parameters,
            )
        parameters['tools'] += tool_definitions
        parameters['tool_choice'] = "auto"

        # Create chat with the arguments
        chat = self.client.chat.create(**parameters)

        chat, _ = self.convert_conversation_history_to_adapter_format(chat, the_conversation, model)
        
        response = chat.sample()

        usage = {"model": model,
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens}
        
        # Save tool_calls parameter from the openai answer for the history
        if getattr(response, 'tool_calls', None):
            function_call_records = [FunctionCall.from_grok(each_tool_call) for each_tool_call in response.tool_calls]
        else: 
            return response # If there are no tool calls, return the response directly

        function_response_records = []
        for each_tools_call in function_call_records:

            tool_call_id = each_tools_call.id
            tool_function_name = each_tools_call.name
            tool_arguments = json.loads(each_tools_call.arguments)

            # Find the requested function
            function_index = next((i for i, tool in enumerate(tool_definitions) if tool.function.name == tool_function_name), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_tools_call.name} not found in tools")
            function = functions[function_index]

            # Get all function parameters
            function_parameters = []
            parameters_from_tool_definition = json.loads(tool_definitions[function_index].function.parameters)
            for key, value in parameters_from_tool_definition["properties"].items():
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
        if response.reasoning_content:
            thinking_responses.append(ThinkingResponse(content=response.reasoning_content, id=response.id))
        
        message = Message(
            role="assistant", 
            content=response.content,
            thinking_responses=thinking_responses,
            function_calls=function_call_records,
            function_responses=function_response_records,
            usage=usage
        )
        the_conversation.messages.append(message)

        final_response = self.request_llm_with_functions(model, the_conversation, functions, tool_output_callback=tool_output_callback, **kwargs)
        
        return final_response


    def voice_to_text(self, audio_file):
        raise NotImplementedError("OpenRoute does not support voice to text")

    def generate_image(self, prompt: str, n: int=1, **kwargs) -> List[ImageFile]:
        response = self.client.image.sample_batch(
                model="grok-2-image",
                prompt=prompt,
                n=n,
                image_format="base64",
            )

        output_images = [ImageFile.from_bytes(file_bytes=image_data.image, file_name="image.png") for image_data in response]

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
        tool_item = tool(
            name = func.__name__,
            description = func.__doc__ or '',
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
        )

        return tool_item

    def _convert_function_to_tool(self, func: BaseTool | Callable) -> Dict:
        
        # Convert the function to a tool for OpenAI
        if isinstance(func, BaseTool):
            # Handle the case where func is a BaseTool
            tool = func.to_params(provider='grok')
        
        elif callable(func):
            tool = self._convert_func_to_tool(func)
        
        else:
            raise TypeError("func must be either a BaseTool or a function")
        
        return tool

    def get_models(self) -> List[str]:
        raise NotImplementedError("Not implemented yet")
