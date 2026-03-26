import inspect
import json
import os
from typing import Callable, Dict, List

from loguru import logger
from xai_sdk.chat import image, system, tool, tool_result
from xai_sdk.proto import chat_pb2
from xai_sdk.tools import code_execution, web_search, x_search

from llm_platform.services.conversation import (
    Conversation,
    FunctionCall,
    FunctionResponse,
    Message,
    ThinkingResponse,
)
from llm_platform.services.files import (
    AudioFile,
    ExcelDocumentFile,
    ImageFile,
    PDFDocumentFile,
    PowerPointDocumentFile,
    TextDocumentFile,
    WordDocumentFile,
)
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters

from .adapter_base import AdapterBase


class GrokAdapter(AdapterBase):
    ROLE_MAPPING = {
        "user": chat_pb2.MessageRole.ROLE_USER,
        "assistant": chat_pb2.MessageRole.ROLE_ASSISTANT,
    }

    def __init__(self):
        super().__init__()
        from xai_sdk import Client

        self.client = Client(api_key=os.getenv("XAI_API_KEY"))

    def _merge_additional_parameters(
        self,
        additional_parameters: AdditionalParameters | None,
        kwargs: Dict,
    ) -> AdditionalParameters:
        merged_parameters = dict(additional_parameters or {})

        if kwargs:
            logger.warning(
                "Passing request parameters via **kwargs is deprecated; use additional_parameters."
            )
            for key, value in kwargs.items():
                merged_parameters.setdefault(key, value)

        return merged_parameters

    def _get_structured_output_model(
        self,
        additional_parameters: AdditionalParameters,
    ):
        structured_output = additional_parameters.get("structured_output")
        if structured_output is None or structured_output is False:
            return None
        if structured_output is True:
            raise ValueError(
                "Grok structured_output must be a Pydantic model class, not a boolean flag."
            )
        return structured_output

    def convert_conversation_history_to_adapter_format(
        self,
        chat,
        the_conversation: Conversation,
        model: str,
        **kwargs,
    ):
        if the_conversation.system_prompt:
            chat.append(system(the_conversation.system_prompt))

        for message in the_conversation.messages:
            role = self.ROLE_MAPPING.get(message.role)
            if role is None:
                raise ValueError(f"Unsupported Grok message role: {message.role}")

            message_parameters = {
                "role": role,
                "content": [],
            }

            if message.content:
                message_parameters["content"].append(chat_pb2.Content(text=message.content))

            if message.function_calls:
                message_parameters["tool_calls"] = [
                    chat_pb2.ToolCall(
                        id=function_call.call_id,
                        function=chat_pb2.FunctionCall(
                            name=function_call.name,
                            arguments=function_call.arguments,
                        ),
                    )
                    for function_call in message.function_calls
                ]

            if message.files:
                for each_file in message.files:
                    if isinstance(each_file, ImageFile):
                        message_parameters["content"].append(
                            image(
                                image_url=f"data:image/{each_file.extension};base64,{each_file.base64}",
                                detail="high",
                            )
                        )
                    elif isinstance(each_file, AudioFile):
                        raise NotImplementedError("Grok does not support audio files")
                    elif isinstance(
                        each_file,
                        (
                            TextDocumentFile,
                            ExcelDocumentFile,
                            PDFDocumentFile,
                            WordDocumentFile,
                            PowerPointDocumentFile,
                        ),
                    ):
                        document_content = chat_pb2.Content(
                            text=f'<document name="{each_file.name}">{each_file.text}</document>'
                        )
                        message_parameters["content"].append(document_content)
                    else:
                        raise ValueError(
                            f"Unsupported file type for Grok: {type(each_file).__name__}"
                        )

            chat.append(chat_pb2.Message(**message_parameters))

            if message.function_responses:
                for function_response in message.function_responses:
                    chat.append(
                        tool_result(
                            result=json.dumps(function_response.response, default=str),
                            tool_call_id=function_response.call_id,
                        )
                    )

        return chat, kwargs

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        additional_parameters: AdditionalParameters | None = None,
    ) -> Dict:
        additional_parameters = additional_parameters or {}

        parameters = {
            "model": model,
            "tools": [],
        }

        reserved_parameters = {
            "web_search",
            "code_execution",
            "response_modalities",
            "citations_enabled",
            "url_context",
            "structured_output",
            "reasoning",
            "text",
            "agent_count",
        }
        for key, value in additional_parameters.items():
            if key in reserved_parameters:
                continue
            parameters[key] = value

        if additional_parameters.get("web_search"):
            parameters["tools"].extend([web_search(), x_search()])

        if additional_parameters.get("code_execution"):
            parameters["tools"].append(code_execution())

        if agent_count := additional_parameters.get("agent_count", None):
            parameters['agent_count'] = int(agent_count)

        if reasoning_parameter := additional_parameters.get("reasoning", {}):
            reasoning_effort = reasoning_parameter.get("effort")
            if reasoning_effort:
                parameters["reasoning_effort"] = reasoning_effort

        if structured_output_model := self._get_structured_output_model(additional_parameters):
            parameters["response_format"] = structured_output_model

            if parameters["tools"] and not model.startswith("grok-4"):
                raise ValueError(
                    "Structured output with tools is only supported for the Grok 4 family."
                )

        return parameters

    def _build_usage(self, response, model: str) -> Dict:
        usage = getattr(response, "usage", None)
        return {
            "model": model,
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
        }

    def _build_thinking_responses(self, response) -> List[ThinkingResponse]:
        if not getattr(response, "reasoning_content", None):
            return []
        return [ThinkingResponse(content=response.reasoning_content, id=response.id)]

    def _build_message_from_response(
        self,
        response,
        model: str,
        function_calls: List[FunctionCall] | None = None,
        function_responses: List[FunctionResponse] | None = None,
    ) -> Message:
        return Message(
            role="assistant",
            id=getattr(response, "id", None),
            content=getattr(response, "content", "") or "",
            thinking_responses=self._build_thinking_responses(response),
            function_calls=function_calls or [],
            function_responses=function_responses or [],
            usage=self._build_usage(response, model),
        )

    def _execute_tool_calls(
        self,
        function_call_records: List[FunctionCall],
        functions: List[BaseTool | Callable],
        tool_definitions: List,
        tool_output_callback: Callable = None,
    ) -> List[FunctionResponse]:
        tool_map = {
            tool_definition.function.name: function
            for tool_definition, function in zip(tool_definitions, functions)
        }

        function_response_records = []
        for function_call in function_call_records:
            function_to_call = tool_map.get(function_call.name)
            if function_to_call is None:
                raise ValueError(f"Function {function_call.name} not found in tools")

            tool_arguments = json.loads(function_call.arguments)
            function_response = function_to_call(**tool_arguments)

            function_response_records.append(
                FunctionResponse(
                    name=function_call.name,
                    id=function_call.id,
                    call_id=function_call.call_id,
                    response=function_response,
                )
            )

            if tool_output_callback:
                tool_output_callback(function_call.name, tool_arguments, function_response)

        return function_response_records

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        if functions:
            response = self.request_llm_with_functions(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )
        else:
            parameters = self._create_parameters_for_calling_llm(
                model=model,
                additional_parameters=additional_parameters,
            )

            chat = self.client.chat.create(**parameters)
            chat, _ = self.convert_conversation_history_to_adapter_format(
                chat, the_conversation, model
            )
            response = chat.sample()

        message = self._build_message_from_response(response=response, model=model)
        the_conversation.messages.append(message)
        return message

    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ):
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        tool_definitions = [self._convert_function_to_tool(function) for function in functions]

        parameters = self._create_parameters_for_calling_llm(
            model=model,
            additional_parameters=additional_parameters,
        )
        parameters["tools"].extend(tool_definitions)
        parameters["tool_choice"] = "auto"

        if self._get_structured_output_model(additional_parameters) and not model.startswith(
            "grok-4"
        ):
            raise ValueError(
                "Structured output with tools is only supported for the Grok 4 family."
            )

        chat = self.client.chat.create(**parameters)
        chat, _ = self.convert_conversation_history_to_adapter_format(
            chat, the_conversation, model
        )
        response = chat.sample()

        if not getattr(response, "tool_calls", None):
            return response

        function_call_records = [
            FunctionCall.from_grok(each_tool_call) for each_tool_call in response.tool_calls
        ]
        function_response_records = self._execute_tool_calls(
            function_call_records=function_call_records,
            functions=functions,
            tool_definitions=tool_definitions,
            tool_output_callback=tool_output_callback,
        )

        the_conversation.messages.append(
            self._build_message_from_response(
                response=response,
                model=model,
                function_calls=function_call_records,
                function_responses=function_response_records,
            )
        )

        return self.request_llm_with_functions(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
        )

    def voice_to_text(self, audio_file):
        raise NotImplementedError("Grok does not support voice to text")

    def generate_image(self, prompt: str, n: int = 1, **kwargs) -> List[ImageFile]:
        """Generates images based on the provided prompt using the old grok-2-image model.
        The current functionality to generate image is in the package `adapters/grok_image_adapter.py`
        This old method can be removed later.
        """
        response = self.client.image.sample_batch(
            model="grok-2-image",
            prompt=prompt,
            n=n,
            image_format="base64",
        )

        output_images = [
            ImageFile.from_bytes(file_bytes=image_data.image, file_name="image.png")
            for image_data in response
        ]

        return output_images

    def _convert_func_to_tool(self, func: Callable) -> Dict:
        sig = inspect.signature(func)

        parameters = {}
        required_params = []

        for param_name, param in sig.parameters.items():
            param_info = {}

            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_info["type"] = "string"
                elif param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list:
                    param_info["type"] = "array"
                elif param.annotation == dict:
                    param_info["type"] = "object"
                else:
                    param_info["type"] = "string"
            else:
                param_info["type"] = "string"

            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)

            parameters[param_name] = param_info

        return tool(
            name=func.__name__,
            description=func.__doc__ or "",
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
        )

    def _convert_function_to_tool(self, func: BaseTool | Callable) -> Dict:
        if isinstance(func, BaseTool):
            tool_definition = func.to_params(provider="grok")
        elif callable(func):
            tool_definition = self._convert_func_to_tool(func)
        else:
            raise TypeError("func must be either a BaseTool or a function")

        return tool_definition

    def get_models(self) -> List[str]:
        raise NotImplementedError("Not implemented yet")
