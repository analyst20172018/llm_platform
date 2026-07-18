import inspect
import json
import os
from typing import Any, Callable, Dict, List

from llm_platform.adapters.serializers import function_call_from_openai_chat
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
    VideoFile,
    WordDocumentFile,
)
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters

from .adapter_base import MAX_TOOL_ROUNDS
from .openai_compatible_adapter import OpenAICompatibleAdapter


class KimiAdapter(OpenAICompatibleAdapter):
    """Moonshot AI's OpenAI-compatible Chat Completions adapter.

    Kimi K3 needs provider-specific handling beyond the shared compatibility
    layer: its output limit is named ``max_completion_tokens``, sampling values
    are fixed, reasoning is returned in ``reasoning_content``, and that reasoning
    must be preserved with the complete assistant message on later turns.
    """

    BASE_URL = "https://api.moonshot.ai/v1"
    ENV_VAR = "MOONSHOT_API_KEY"

    _FIXED_SAMPLING_PARAMETERS = {
        "temperature",
        "top_p",
        "n",
        "presence_penalty",
        "frequency_penalty",
    }

    @classmethod
    def _api_key(cls) -> str:
        api_key = os.getenv(cls.ENV_VAR)
        if not api_key:
            raise ValueError(f"{cls.ENV_VAR} is required to call Kimi models")
        return api_key

    def _build_client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.BASE_URL, api_key=self._api_key())

    def _build_async_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=self.BASE_URL, api_key=self._api_key())

    def convert_conversation_history_to_adapter_format(
        self, the_conversation: Conversation, model: str, **kwargs
    ):
        history = [{"role": "system", "content": the_conversation.system_prompt}]

        for message in the_conversation.messages:
            history_message: Dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }

            if message.thinking_responses and message.role == "assistant":
                history_message["reasoning_content"] = "\n".join(
                    response.content for response in message.thinking_responses
                )

            if message.function_calls:
                history_message["tool_calls"] = [
                    {
                        "id": function_call.call_id,
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": function_call.arguments,
                        },
                    }
                    for function_call in message.function_calls
                ]

            for each_file in message.files:
                if isinstance(each_file, ImageFile):
                    self._append_content_part(
                        history_message,
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(each_file)},
                        },
                    )
                elif isinstance(each_file, VideoFile):
                    self._require_model_input(model, "video")
                    self._append_content_part(
                        history_message,
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": (
                                    f"data:video/{each_file.extension};base64,"
                                    f"{each_file.base64}"
                                )
                            },
                        },
                    )
                elif isinstance(each_file, AudioFile):
                    raise ValueError(f"Model {model} does not support audio input.")
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
                    self._prepend_content_part(
                        history_message,
                        {"type": "text", "text": self._document_xml(each_file)},
                    )
                else:
                    raise ValueError(
                        f"Unsupported file type in file {each_file.name}. "
                        f"The type is {type(each_file)}."
                    )

            history.append(history_message)
            history.extend(
                {
                    "role": "tool",
                    "tool_call_id": response.call_id,
                    "content": json.dumps(response.response),
                }
                for response in message.function_responses
            )

        return history, kwargs

    @staticmethod
    def _append_content_part(message: Dict[str, Any], part: Dict[str, Any]) -> None:
        if not isinstance(message["content"], list):
            message["content"] = [{"type": "text", "text": message["content"]}]
        message["content"].append(part)

    @staticmethod
    def _prepend_content_part(message: Dict[str, Any], part: Dict[str, Any]) -> None:
        if not isinstance(message["content"], list):
            message["content"] = [{"type": "text", "text": message["content"]}]
        message["content"].insert(0, part)

    def _require_model_input(self, model: str, input_type: str) -> None:
        model_object = self.model_config[model]
        if not (model_object and input_type in model_object.inputs):
            raise ValueError(f"Model {model} does not support {input_type} input.")

    def _build_request_params(
        self,
        model: str,
        additional_parameters: AdditionalParameters,
    ) -> Dict[str, Any]:
        request_params = super()._build_request_params(model, additional_parameters)

        raw_max_tokens = request_params.pop("max_tokens", None)
        if raw_max_tokens is not None:
            request_params.setdefault("max_completion_tokens", raw_max_tokens)

        for parameter in self._FIXED_SAMPLING_PARAMETERS:
            request_params.pop(parameter, None)

        structured_output = additional_parameters.get("structured_output")
        if structured_output:
            request_params["response_format"] = self._structured_output_format(
                structured_output
            )

        return request_params

    @staticmethod
    def _structured_output_format(structured_output: Any) -> Dict[str, Any]:
        if isinstance(structured_output, dict) and structured_output.get("type") in {
            "json_object",
            "json_schema",
        }:
            return structured_output

        if hasattr(structured_output, "model_json_schema"):
            schema = structured_output.model_json_schema()
            name = structured_output.__name__
        elif isinstance(structured_output, dict):
            schema = structured_output
            name = str(schema.get("title", "response"))
        else:
            raise TypeError(
                "structured_output must be a Pydantic model class or a JSON Schema dict"
            )

        return {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        }

    def _convert_function_to_tool(
        self, function: BaseTool | Callable
    ) -> Dict[str, Any]:
        if isinstance(function, BaseTool):
            schema = function.to_params(provider="openai")
        elif callable(function):
            schema = self._callable_to_json_schema(function)
        else:
            raise TypeError("function must be a BaseTool or callable")
        return {"type": "function", "function": schema}

    @staticmethod
    def _function_calls_from_response(assistant_message) -> List[FunctionCall]:
        return [
            function_call_from_openai_chat(tool_call)
            for tool_call in (getattr(assistant_message, "tool_calls", None) or [])
        ]

    @staticmethod
    def _thinking_from_response(response, assistant_message) -> List[ThinkingResponse]:
        reasoning_content = getattr(assistant_message, "reasoning_content", None)
        if not reasoning_content:
            return []
        return [
            ThinkingResponse(
                content=reasoning_content, id=getattr(response, "id", None)
            )
        ]

    def _execute_tool_calls(
        self,
        function_calls: List[FunctionCall],
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable | None,
    ) -> List[FunctionResponse]:
        function_map = {self._tool_name(function): function for function in functions}
        responses = []
        for function_call in function_calls:
            function = function_map.get(function_call.name)
            if function is None:
                raise ValueError(f"Function {function_call.name} not found in tools")
            arguments = json.loads(function_call.arguments)
            result = function(**arguments)
            responses.append(
                FunctionResponse(
                    name=function_call.name,
                    call_id=function_call.call_id,
                    response=result,
                )
            )
            if tool_output_callback:
                tool_output_callback(function_call.name, arguments, result)
        return responses

    async def _execute_tool_calls_async(
        self,
        function_calls: List[FunctionCall],
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable | None,
    ) -> List[FunctionResponse]:
        function_map = {self._tool_name(function): function for function in functions}
        responses = []
        for function_call in function_calls:
            function = function_map.get(function_call.name)
            if function is None:
                raise ValueError(f"Function {function_call.name} not found in tools")
            arguments = json.loads(function_call.arguments)
            result = function(**arguments)
            if inspect.isawaitable(result):
                result = await result
            responses.append(
                FunctionResponse(
                    name=function_call.name,
                    call_id=function_call.call_id,
                    response=result,
                )
            )
            if tool_output_callback:
                tool_output_callback(function_call.name, arguments, result)
        return responses

    def _record_response(
        self,
        model: str,
        response,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable | None,
    ) -> Message | None:
        assistant_message = response.choices[0].message
        function_calls = self._function_calls_from_response(assistant_message)
        thinking_responses = self._thinking_from_response(response, assistant_message)
        usage = self._build_usage(getattr(response, "usage", None), model)

        if not function_calls:
            message = Message(
                role="assistant",
                content=assistant_message.content or "",
                thinking_responses=thinking_responses,
                usage=usage,
            )
            the_conversation.messages.append(message)
            return message

        function_responses = self._execute_tool_calls(
            function_calls, functions, tool_output_callback
        )
        the_conversation.messages.append(
            Message(
                role="assistant",
                content=assistant_message.content or "",
                thinking_responses=thinking_responses,
                function_calls=function_calls,
                function_responses=function_responses,
                usage=usage,
            )
        )
        return None

    async def _record_response_async(
        self,
        model: str,
        response,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable | None,
    ) -> Message | None:
        assistant_message = response.choices[0].message
        function_calls = self._function_calls_from_response(assistant_message)
        thinking_responses = self._thinking_from_response(response, assistant_message)
        usage = self._build_usage(getattr(response, "usage", None), model)

        if not function_calls:
            message = Message(
                role="assistant",
                content=assistant_message.content or "",
                thinking_responses=thinking_responses,
                usage=usage,
            )
            the_conversation.messages.append(message)
            return message

        function_responses = await self._execute_tool_calls_async(
            function_calls, functions, tool_output_callback
        )
        the_conversation.messages.append(
            Message(
                role="assistant",
                content=assistant_message.content or "",
                thinking_responses=thinking_responses,
                function_calls=function_calls,
                function_responses=function_responses,
                usage=usage,
            )
        )
        return None

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        functions = list(functions or [])
        request_params = self._build_request_params(model, parameters)
        if functions:
            request_params["tools"] = [
                self._convert_function_to_tool(function) for function in functions
            ]
            request_params.setdefault("tool_choice", "auto")
        else:
            request_params.pop("tool_choice", None)

        for _ in range(MAX_TOOL_ROUNDS):
            history, history_kwargs = (
                self.convert_conversation_history_to_adapter_format(
                    the_conversation, model
                )
            )
            response = self.client.chat.completions.create(
                model=model,
                messages=history,
                **request_params,
                **history_kwargs,
            )
            message = self._record_response(
                model,
                response,
                the_conversation,
                functions,
                tool_output_callback,
            )
            if message is not None:
                return message

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

    async def request_llm_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        functions = list(functions or [])
        request_params = self._build_request_params(model, parameters)
        if functions:
            request_params["tools"] = [
                self._convert_function_to_tool(function) for function in functions
            ]
            request_params.setdefault("tool_choice", "auto")
        else:
            request_params.pop("tool_choice", None)

        for _ in range(MAX_TOOL_ROUNDS):
            history, history_kwargs = (
                self.convert_conversation_history_to_adapter_format(
                    the_conversation, model
                )
            )
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=history,
                **request_params,
                **history_kwargs,
            )
            message = await self._record_response_async(
                model,
                response,
                the_conversation,
                functions,
                tool_output_callback,
            )
            if message is not None:
                return message

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )
