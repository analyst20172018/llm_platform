import asyncio
import copy
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, List

from llm_platform.services.conversation import Conversation, FunctionResponse, Message, ThinkingResponse
from llm_platform.services.files import (
    AudioFile,
    ExcelDocumentFile,
    ImageFile,
    PDFDocumentFile,
    PowerPointDocumentFile,
    TextDocumentFile,
    WordDocumentFile,
    VideoFile,
)
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import (
    function_call_to_openai_chat,
    function_response_to_openai_chat,
    function_call_from_openai_chat,
    chat_replay_data,
)
from llm_platform.types import AdditionalParameters

from .response_metadata import chat_metadata, block_metadata
from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS
from .json_output import response_format, add_json_instruction

# Platform-level keys consumed by the facade / handled explicitly here, so they
# are never forwarded verbatim to the OpenAI-compatible chat completions call.
OPENAI_COMPATIBLE_RESERVED_KEYS = {
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


class OpenAICompatibleAdapter(AdapterBase):
    """Shared adapter for providers exposing an OpenAI-compatible Chat Completions API.

    Subclasses only declare ``BASE_URL`` and ``ENV_VAR`` (and optionally override
    ``_suppress_temperature``); everything else — sync and async client
    construction, conversation conversion, parameter marshalling, usage
    extraction — is shared here. Subclasses opt into local tools and JSON output.
    """

    BASE_URL: str = None
    ENV_VAR: str = None
    SUPPORTS_TOOLS = False
    JSON_OUTPUT_MODE = None

    @property
    def provider(self):
        return type(self).__name__.removesuffix("Adapter").lower()

    def __init__(self):
        super().__init__()
        self._async_client = None

    def _build_client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.BASE_URL, api_key=os.getenv(self.ENV_VAR))

    def _build_async_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=self.BASE_URL, api_key=os.getenv(self.ENV_VAR))

    @property
    def async_client(self):
        """Async SDK client, constructed lazily once on first access."""
        if self._async_client is None:
            self._async_client = self._build_async_client()
        return self._async_client

    def _suppress_temperature(self, model: str) -> bool:
        """Whether to drop the ``temperature`` parameter for a given model.

        Driven by the per-model ``suppress_temperature`` flag in
        models_config.yaml. No model currently sets it; it is a dormant per-model
        extension point for any future model that rejects ``temperature``.
        """
        model_object = self.model_config[model]
        return bool(model_object and model_object["suppress_temperature"])

    def convert_conversation_history_to_adapter_format(
        self, the_conversation: Conversation, model: str, **kwargs
    ):
        # System prompt is the first message
        history = [{"role": "system", "content": the_conversation.system_prompt}]

        for message in the_conversation.messages:
            if message.role == "function":
                history.extend(function_response_to_openai_chat(fr) for fr in message.function_responses)
                continue
            history_message = {"role": message.role, "content": message.content}

            if message.function_calls:
                history_message["tool_calls"] = [
                    function_call_to_openai_chat(each_call) for each_call in message.function_calls
                ]

            if message.files is not None:
                for each_file in message.files:
                    if isinstance(each_file, ImageFile):
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{"type": "text", "text": history_message["content"]}]
                        history_message["content"].append(
                            {"type": "image_url", "image_url": {"url": self._image_data_url(each_file)}}
                        )

                    elif isinstance(each_file, VideoFile):
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{"type": "text", "text": history_message["content"]}]
                        history_message["content"].append(self._video_content(each_file, model))

                    elif isinstance(each_file, AudioFile):
                        model_object = self.model_config[model]
                        if not (model_object and "audio" in model_object.inputs):
                            raise ValueError(f"Model {model} does not support audio input.")
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{"type": "text", "text": history_message["content"]}]
                        # AudioFile converts its payload to mp3 on construction, so the
                        # declared format is always mp3 regardless of the original extension.
                        history_message["content"].append(
                            {"type": "input_audio", "input_audio": {"data": each_file.base64, "format": "mp3"}}
                        )

                    elif isinstance(
                        each_file,
                        (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile),
                    ):
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{"type": "text", "text": history_message["content"]}]
                        history_message["content"].insert(0, self._document_content(each_file, model))

                    else:
                        raise ValueError(
                            f"Unsupported file type in file {each_file.name}. The type is {type(each_file)}."
                        )

            native = message.replay_data(self.provider, model)
            history.append(native.get("message", history_message))

            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(function_response_to_openai_chat(each_response))

        return history, kwargs

    def _video_content(self, file, model):
        raise ValueError(f"Model {model} does not support video input.")

    def _document_content(self, file, model):
        return {"type": "text", "text": self._document_xml(file)}

    def _build_request_params(self, model: str, additional_parameters: AdditionalParameters) -> Dict[str, Any]:
        request_params: Dict[str, Any] = {}
        if "temperature" in additional_parameters and not self._suppress_temperature(model):
            request_params["temperature"] = additional_parameters["temperature"]
        if "max_tokens" in additional_parameters:
            request_params["max_tokens"] = additional_parameters["max_tokens"]

        for key, value in additional_parameters.items():
            if key in OPENAI_COMPATIBLE_RESERVED_KEYS:
                continue
            request_params[key] = copy.deepcopy(value)

        if additional_parameters.get("structured_output") and self.JSON_OUTPUT_MODE:
            output_format = response_format(additional_parameters["structured_output"])
            request_params["response_format"] = (
                {"type": "json_object"} if self.JSON_OUTPUT_MODE == "json_object" else output_format
            )

        return request_params

    def _prepare_history(self, conversation, model, structured_output=None):
        history, kwargs = self.convert_conversation_history_to_adapter_format(conversation, model)
        if structured_output and self.JSON_OUTPUT_MODE:
            add_json_instruction(history, response_format(structured_output))
        return history, kwargs

    def _prepare_tools(self, model, functions, request_params):
        if not functions:
            return {}
        model_object = self.model_config[model]
        if not self.SUPPORTS_TOOLS or not (
            model_object and model_object.get_parameter("function_calling")
        ):
            raise NotImplementedError(f"{type(self).__name__} does not support tool calling for {model}")
        tools, tool_map = [], {}
        for function in functions:
            if isinstance(function, BaseTool):
                schema = function.to_params(provider="openai")
            elif callable(function):
                schema = self._callable_to_json_schema(function)
            else:
                raise TypeError("function must be a BaseTool or callable")
            name = schema["name"]
            if name in tool_map:
                raise ValueError(f"Duplicate tool name: {name}")
            tool_map[name] = function
            tools.append({"type": "function", "function": schema})
        request_params["tools"] = tools
        request_params.setdefault("tool_choice", "auto")
        return tool_map

    @staticmethod
    def _tool_invocation(call, tool_map):
        if call.name not in tool_map:
            raise ValueError(f"Function {call.name} not found in tools")
        arguments = json.loads(call.arguments)
        if not isinstance(arguments, dict):
            raise ValueError(f"Arguments for {call.name} must be a JSON object")
        return tool_map[call.name], arguments

    @staticmethod
    def _record_tool_result(conversation, call, result):
        conversation.messages.append(Message(role="function", content="", function_responses=[
            FunctionResponse(name=call.name, id=call.id, call_id=call.call_id, response=result)
        ]))

    def _message_from_response(self, model: str, response) -> Message:
        """Build the assistant message, capturing reasoning when the provider returns it.

        Reasoning models on OpenAI-compatible endpoints (e.g. DeepSeek) return the
        chain of thought in ``reasoning_content``, alongside ``content``.
        """
        assistant_message = response.choices[0].message
        reasoning_content = (getattr(assistant_message, "reasoning_content", None)
                             or getattr(assistant_message, "reasoning", None))
        thinking_responses = (
            [ThinkingResponse(content=reasoning_content, id=getattr(response, "id", None))]
            if reasoning_content
            else []
        )
        return Message(
            **chat_metadata(response),
            id=getattr(response, "id", None),
            provider=self.provider, model=model,
            provider_data={"message": chat_replay_data(assistant_message)},
            **block_metadata(self.provider, [chat_replay_data(assistant_message)]),
            role="assistant",
            content=assistant_message.content or "",
            function_calls=[function_call_from_openai_chat(call)
                            for call in (getattr(assistant_message, "tool_calls", None) or [])],
            thinking_responses=thinking_responses,
            usage=self._build_usage(getattr(response, "usage", None), model),
        )

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

        request_params = self._build_request_params(model, additional_parameters)
        tool_map = self._prepare_tools(model, functions, request_params)
        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        for _ in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline)
            history, history_kwargs = self._prepare_history(
                the_conversation, model, additional_parameters.get("structured_output")
            )
            response = self.client.chat.completions.create(
                model=model, messages=history, **{**request_params, **history_kwargs}
            )
            message = self._message_from_response(model, response)
            the_conversation.messages.append(message)
            if not tool_map or not message.can_execute_tools or not message.function_calls:
                return message
            for call in message.function_calls:
                self._check_deadline(deadline, response)
                function, arguments = self._tool_invocation(call, tool_map)
                result = function(**arguments)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("Async tools require request_llm_async")
                self._record_tool_result(the_conversation, call, result)
                if tool_output_callback:
                    tool_output_callback(call.name, arguments, result)
            request_params["tool_choice"] = "auto"
        raise RuntimeError(f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}")

    def request_llm_with_functions(self, *args, **kwargs):
        return self.request_llm(*args, **kwargs)

    async def request_llm_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        """Async counterpart of `request_llm`, backed by the native `AsyncOpenAI` client."""
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        request_params = self._build_request_params(model, additional_parameters)
        tool_map = self._prepare_tools(model, functions, request_params)
        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        for _ in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline)
            history, history_kwargs = self._prepare_history(
                the_conversation, model, additional_parameters.get("structured_output")
            )
            response = await self.async_client.chat.completions.create(
                model=model, messages=history, **{**request_params, **history_kwargs}
            )
            message = self._message_from_response(model, response)
            the_conversation.messages.append(message)
            if not tool_map or not message.can_execute_tools or not message.function_calls:
                return message
            for call in message.function_calls:
                self._check_deadline(deadline, response)
                function, arguments = self._tool_invocation(call, tool_map)
                if inspect.iscoroutinefunction(function):
                    result = await function(**arguments)
                else:
                    result = await asyncio.to_thread(function, **arguments)
                    if inspect.isawaitable(result):
                        result = await result
                self._record_tool_result(the_conversation, call, result)
                if tool_output_callback:
                    callback_result = tool_output_callback(call.name, arguments, result)
                    if inspect.isawaitable(callback_result):
                        await callback_result
            request_params["tool_choice"] = "auto"
        raise RuntimeError(f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}")
