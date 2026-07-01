from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS
import os
from typing import Any, Callable, Dict, List
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import (function_call_from_openai_chat,
                                                  function_call_to_openai_chat,
                                                  function_response_to_openai_chat)
from llm_platform.services.conversation import Conversation, Message, FunctionResponse
from llm_platform.services.files import (AudioFile, TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile,
                                         ImageFile)
import json
from llm_platform.types import AdditionalParameters

# Platform-level keys consumed by the facade / handled explicitly here, so they
# are never forwarded verbatim to the Mistral chat completions call.
MISTRAL_RESERVED_KEYS = {
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

            # Tool calls are serialized in Chat Completions shape (tool_calls[].function).
            if message.function_calls:
                history_message["tool_calls"] = [function_call_to_openai_chat(each_call) for each_call in message.function_calls]

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
                    elif isinstance(each_file, AudioFile):
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
                        raise ValueError(f"Unsupported file type: {type(each_file).__name__} in file {each_file.name}")

            history.append(history_message)

            # Tool results are sent as standalone `role: "tool"` messages.
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(function_response_to_openai_chat(each_response))

        return history, kwargs

    def _build_request_params(self, additional_parameters: AdditionalParameters) -> Dict[str, Any]:
        request_params: Dict[str, Any] = {}
        if "temperature" in additional_parameters:
            request_params["temperature"] = additional_parameters["temperature"]
        if "max_tokens" in additional_parameters:
            request_params["max_tokens"] = additional_parameters["max_tokens"]

        for key, value in additional_parameters.items():
            if key in MISTRAL_RESERVED_KEYS:
                continue
            request_params[key] = value

        return request_params

    def request_llm(self, model: str,
                    the_conversation: Conversation,
                    functions:List[BaseTool]=None,
                    tool_output_callback: Callable=None,
                    additional_parameters: AdditionalParameters | None = None,
                    **kwargs) -> Message:

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        # LLM with functions
        if functions:
            return self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            additional_parameters=additional_parameters,
                        )

        # Standard text LLM
        request_params = self._build_request_params(additional_parameters)
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
                                   _tool_round: int = 0,
                                   **kwargs) -> Message:
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        request_params = self._build_request_params(additional_parameters)

        tools = [self._convert_function_to_tool(each_function) for each_function in functions]
        messages, _ = self.convert_conversation_history_to_adapter_format(the_conversation, model)

        chat_response = self.client.chat.complete(
            model = model,
            messages = messages,
            tools = tools,
            tool_choice = "auto",
            **request_params,
        )

        usage = self._build_usage(getattr(chat_response, "usage", None), model)

        assistant_message = chat_response.choices[0].message
        tool_calls = getattr(assistant_message, 'tool_calls', None)

        # No tool calls -> final answer; record it and finish.
        if not tool_calls:
            message = Message(role="assistant", content=assistant_message.content, usage=usage)
            the_conversation.messages.append(message)
            return message

        function_call_records = [function_call_from_openai_chat(each_tool_call) for each_tool_call in tool_calls]

        function_response_records = []
        for each_call in function_call_records:

            tool_arguments = json.loads(each_call.arguments)

            # Find the requested function
            function_index = next((i for i, tool in enumerate(tools) if tool['function']['name'] == each_call.name), -1)
            if function_index == -1:
                raise ValueError(f"Function {each_call.name} not found in tools")
            function = functions[function_index]

            # Call the function with keyword arguments (robust to optional/reordered args,
            # matching the OpenAI/Grok adapters and avoiding KeyError on omitted optionals).
            function_response = function(**tool_arguments)

            function_response_record = FunctionResponse(name=each_call.name,
                                                        id=each_call.id,
                                                        call_id=each_call.call_id,
                                                        response=function_response)
            function_response_records.append(function_response_record)

            if tool_output_callback:
                tool_output_callback(each_call.name,
                                     tool_arguments,
                                     function_response
                                    )

        message = Message(role="assistant",
                            content=assistant_message.content or "",
                            function_calls=function_call_records,
                            function_responses=function_response_records,
                            usage=usage
                            )
        the_conversation.messages.append(message)

        return self.request_llm_with_functions(model,
                                               the_conversation,
                                               functions,
                                               tool_output_callback=tool_output_callback,
                                               additional_parameters=additional_parameters,
                                               _tool_round=_tool_round + 1)
