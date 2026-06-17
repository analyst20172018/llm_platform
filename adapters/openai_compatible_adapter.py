import os
from typing import Any, Callable, Dict, List

from llm_platform.services.conversation import Conversation, Message
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
from llm_platform.adapters.serializers import (
    function_call_to_openai_chat,
    function_response_to_openai_chat,
)
from llm_platform.types import AdditionalParameters

from .adapter_base import AdapterBase

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
    ``_suppress_temperature``); everything else — client construction, conversation
    conversion, parameter marshalling, usage extraction — is shared here. Tool
    calling is not supported (inherits the base ``request_llm_with_functions``).
    """

    BASE_URL: str = None
    ENV_VAR: str = None

    def _build_client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.BASE_URL, api_key=os.getenv(self.ENV_VAR))

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
            history_message = {"role": message.role, "content": message.content}

            if message.function_calls:
                history_message["tool_calls"] = [
                    function_call_to_openai_chat(each_call) for each_call in message.function_calls
                ]

            if message.files is not None:
                for each_file in message.files:
                    if isinstance(each_file, ImageFile):
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]
                        history_message["content"].append(
                            {"type": "image_url", "image_url": {"url": self._image_data_url(each_file)}}
                        )

                    elif isinstance(each_file, AudioFile):
                        model_object = self.model_config[model]
                        if not (model_object and "audio" in model_object.inputs):
                            raise ValueError(f"Model {model} does not support audio input.")
                        kwargs.setdefault("modalities", ["text"])
                        kwargs.setdefault("audio", {"voice": "alloy", "format": "wav"})
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]
                        history_message["content"].append(
                            {"type": "input_audio", "input_audio": {"data": each_file.base64, "format": "mp3"}}
                        )

                    elif isinstance(
                        each_file,
                        (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile),
                    ):
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{"type": "text", "text": history_message["content"]}]
                        history_message["content"].insert(0, {"type": "text", "text": self._document_xml(each_file)})

                    else:
                        raise ValueError(
                            f"Unsupported file type in file {each_file.name}. The type is {type(each_file)}."
                        )

            history.append(history_message)

            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(function_response_to_openai_chat(each_response))

        return history, kwargs

    def _build_request_params(self, model: str, additional_parameters: AdditionalParameters) -> Dict[str, Any]:
        request_params: Dict[str, Any] = {}
        if "temperature" in additional_parameters and not self._suppress_temperature(model):
            request_params["temperature"] = additional_parameters["temperature"]
        if "max_tokens" in additional_parameters:
            request_params["max_tokens"] = additional_parameters["max_tokens"]

        for key, value in additional_parameters.items():
            if key in OPENAI_COMPATIBLE_RESERVED_KEYS:
                continue
            request_params[key] = value

        return request_params

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
            raise NotImplementedError(f"{type(self).__name__} does not support tool calling")

        request_params = self._build_request_params(model, additional_parameters)
        history, history_kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model)
        request_params.update(history_kwargs)

        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            **request_params,
        )

        usage = self._build_usage(getattr(response, "usage", None), model)
        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        return message
