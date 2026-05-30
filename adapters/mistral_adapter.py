from .adapter_base import AdapterBase
import os
from typing import Any, Callable, Dict, List
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import (function_call_from_openai,
                                                  function_call_to_openai,
                                                  function_response_to_openai)
from llm_platform.services.conversation import Conversation, Message, FunctionResponse
from llm_platform.services.files import (AudioFile, TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile,
                                         ImageFile)
import json
from llm_platform.types import AdditionalParameters

class MistralAdapter(AdapterBase):
    
    def _build_client(self):
        # Import lazily so this SDK does not slow down module import time.
        from mistralai.client import Mistral
        return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

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
                history_message["tool_calls"] = [function_call_to_openai(each_call) for each_call in message.function_calls]

            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Add the image to the content list
                        image_content = {"type": "image_url",
                                         "image_url": {"url": self._image_data_url(each_file)}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    if isinstance(each_file, AudioFile):
                        audio_content = {"type": "input_audio", 
                                         "input_audio": each_file.base64
                        }
                        history_message["content"].append(audio_content)
                    
                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile)):

                        # Add the text document to the history as a text in XML tags
                        new_text_content = {
                            "type": "text",
                            "text": self._document_xml(each_file)
                        }
                        history_message["content"].insert(0, new_text_content)

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            history.append(history_message)

            # Add all function responses to the history
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(function_response_to_openai(each_response))

        return history, kwargs

    def request_llm(self, model: str,
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    tool_output_callback: Callable=None, 
                    additional_parameters: AdditionalParameters | None = None,
                    **kwargs) -> Message:

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        request_params: Dict[str, Any] = {}
        if "temperature" in additional_parameters:
            request_params["temperature"] = additional_parameters["temperature"]
        if "max_tokens" in additional_parameters:
            request_params["max_tokens"] = additional_parameters["max_tokens"]

        reserved = {
            "response_modalities",
            "web_search",
            "code_execution",
            "citations_enabled",
            "url_context",
            "structured_output",
            "reasoning",
            "text",
            "temperature",
            "max_tokens",
        }
        for key, value in additional_parameters.items():
            if key in reserved:
                continue
            request_params[key] = value

        # LLM with functions
        if not functions is None:
            response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            **request_params,
                        )
            
        # Standard text LLM
        else:
            messages, history_kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model)
            request_params.update(history_kwargs)
            response = self.client.chat.complete(
                            model=model,
                            messages=messages,
                            **request_params,
                            )
        
        usage = self._build_usage(getattr(response, "usage", None), model)

        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        
        return message
    
    def _convert_func_to_tool(self, func: Callable) -> Dict:
        schema = self._callable_to_json_schema(func)
        return {
            'type': 'function',
            'function': {
                'name': schema['name'],
                'description': schema['description'],
                'parameters': {**schema['parameters'], "additionalProperties": False},
                "strict": True,
            },
        }

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
                                   additional_parameters: AdditionalParameters | None = None,
                                   **kwargs):
        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        messages, _ = self.convert_conversation_history_to_adapter_format(the_conversation, model)

        chat_response = self.client.chat.complete(
            model = model,
            messages = messages,
            tools = tools,
            tool_choice = "any",
        )

        usage = self._build_usage(getattr(chat_response, "usage", None), model)

        assistant_message = chat_response.choices[0].message

        # Save tool_calls parameter from the openai answer for the history
        if getattr(assistant_message, 'tool_calls', None) is not None:
            function_call_records = [function_call_from_openai(each_tool_call) for each_tool_call in assistant_message.tool_calls]
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

            # Call the function with keyword arguments (robust to optional/reordered args,
            # matching the OpenAI/Grok adapters and avoiding KeyError on omitted optionals).
            function_response = function(**tool_arguments)

            function_response_record = FunctionResponse(name=tool_function_name,
                                                        id=tool_call_id,
                                                        response=function_response)
            function_response_records.append(function_response_record)

            if tool_output_callback:
                tool_output_callback(tool_function_name,
                                     tool_arguments,
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
