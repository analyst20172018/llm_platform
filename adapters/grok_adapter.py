import io
import inspect
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

from loguru import logger
import requests
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

GROK_STT_MODEL = "grok-stt"
GROK_TRANSCRIPTION_MODELS = {GROK_STT_MODEL}
GROK_STT_ENDPOINT = "https://api.x.ai/v1/stt"
GROK_STT_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024
GROK_RAW_AUDIO_FORMATS = {"pcm", "mulaw", "alaw"}


class GrokAdapter(AdapterBase):
    ROLE_MAPPING = {
        "user": chat_pb2.MessageRole.ROLE_USER,
        "assistant": chat_pb2.MessageRole.ROLE_ASSISTANT,
    }

    def __init__(self):
        super().__init__()
        from xai_sdk import Client

        self.api_key = os.getenv("XAI_API_KEY")
        self.client = Client(api_key=self.api_key)

    @staticmethod
    def _is_transcription_model(model: str) -> bool:
        return model in GROK_TRANSCRIPTION_MODELS

    @staticmethod
    def _error_message(the_conversation: Conversation, content: str) -> Message:
        message = Message(role="assistant", content=content, usage={})
        the_conversation.messages.append(message)
        return message

    def _extract_audio_file_for_transcription(
        self,
        the_conversation: Conversation,
    ) -> AudioFile | None:
        files = getattr(the_conversation.messages[-1], "files", [])
        if len(files) != 1:
            return None

        audio_file = files[0]
        if not isinstance(audio_file, AudioFile):
            return None

        return audio_file

    @staticmethod
    def _prepare_audio_file_for_transcription(
        audio_source,
    ) -> tuple[str, io.BytesIO, str]:
        if isinstance(audio_source, AudioFile):
            buffer = io.BytesIO(audio_source.file_bytes)
            stem = Path(audio_source.name or "audio").stem or "audio"
            buffer.name = f"{stem}.mp3"
            buffer.seek(0)
            return buffer.name, buffer, "audio/mpeg"

        if hasattr(audio_source, "seek"):
            try:
                audio_source.seek(0)
            except (AttributeError, OSError, ValueError):
                pass

        if hasattr(audio_source, "read"):
            file_bytes = audio_source.read()
            original_name = getattr(audio_source, "name", "") or "audio"
            suffix = Path(original_name).suffix or ".mp3"
            file_name = f"{Path(original_name).stem or 'audio'}{suffix}"
            mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
            buffer = io.BytesIO(file_bytes)
            buffer.name = file_name
            buffer.seek(0)
            return file_name, buffer, mime_type

        raise ValueError("Audio file must be a file-like object or an AudioFile instance.")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _join_transcription_tokens(tokens: List[str]) -> str:
        text = ""
        for token in tokens:
            if not token:
                continue
            if not text:
                text = token
                continue
            if token[0] in ".,!?;:%)]}":
                text += token
                continue
            text += f" {token}"
        return text.strip()

    def _format_diarized_transcription(self, words: List[Dict[str, Any]]) -> str:
        if not words:
            return ""

        segments: List[str] = []
        current_speaker = None
        current_tokens: List[str] = []

        for word in words:
            speaker = word.get("speaker")
            token = str(word.get("text", "")).strip()
            if speaker is None or not token:
                continue

            if current_speaker is None:
                current_speaker = speaker

            if speaker != current_speaker:
                segment_text = self._join_transcription_tokens(current_tokens)
                if segment_text:
                    segments.append(f"[Speaker {current_speaker}]: {segment_text}")
                current_speaker = speaker
                current_tokens = [token]
                continue

            current_tokens.append(token)

        segment_text = self._join_transcription_tokens(current_tokens)
        if segment_text and current_speaker is not None:
            segments.append(f"[Speaker {current_speaker}]: {segment_text}")

        return "\n\n".join(segments)

    def _format_multichannel_transcription(
        self,
        channels: List[Dict[str, Any]],
        diarize: bool,
    ) -> str:
        formatted_channels = []
        for channel in channels:
            index = channel.get("index")
            header = f"Channel {index + 1}" if isinstance(index, int) else "Channel"
            if diarize:
                channel_text = self._format_diarized_transcription(channel.get("words", []))
            else:
                channel_text = str(channel.get("text", "")).strip()

            if channel_text:
                formatted_channels.append(f"{header}:\n{channel_text}")

        return "\n\n".join(formatted_channels)

    def _format_transcription_response(
        self,
        response_payload: Dict[str, Any],
        additional_parameters: AdditionalParameters,
    ) -> str:
        diarize = self._as_bool(additional_parameters.get("diarize"))
        multichannel = self._as_bool(additional_parameters.get("multichannel"))

        if multichannel and response_payload.get("channels"):
            formatted_channels = self._format_multichannel_transcription(
                response_payload["channels"],
                diarize=diarize,
            )
            if formatted_channels:
                return formatted_channels

        if diarize and response_payload.get("words"):
            formatted_diarized = self._format_diarized_transcription(response_payload["words"])
            if formatted_diarized:
                return formatted_diarized

        return str(response_payload.get("text", "")).strip()

    def _build_transcription_request(
        self,
        audio_file,
        additional_parameters: AdditionalParameters,
    ) -> tuple[List[tuple[str, str]], Dict[str, tuple[str, io.BytesIO, str]]]:
        additional_parameters = additional_parameters or {}
        data: List[tuple[str, str]] = []

        language = additional_parameters.get("language")
        if language:
            data.append(("language", str(language)))

        if additional_parameters.get("format") is not None:
            data.append(("format", str(self._as_bool(additional_parameters["format"])).lower()))

        if additional_parameters.get("multichannel") is not None:
            data.append(
                ("multichannel", str(self._as_bool(additional_parameters["multichannel"])).lower())
            )

        if additional_parameters.get("channels") is not None:
            data.append(("channels", str(int(additional_parameters["channels"]))))

        if additional_parameters.get("diarize") is not None:
            data.append(("diarize", str(self._as_bool(additional_parameters["diarize"])).lower()))

        audio_format = additional_parameters.get("audio_format")
        sample_rate = additional_parameters.get("sample_rate")
        if audio_format:
            data.append(("audio_format", str(audio_format)))
            if str(audio_format) in GROK_RAW_AUDIO_FORMATS and sample_rate is None:
                raise ValueError(
                    "Grok STT requires sample_rate when audio_format is pcm, mulaw, or alaw."
                )

        if sample_rate is not None:
            data.append(("sample_rate", str(int(sample_rate))))

        if self._as_bool(additional_parameters.get("format")) and not language:
            raise ValueError("Grok STT requires language when format=true.")

        file_name, buffer, mime_type = self._prepare_audio_file_for_transcription(audio_file)
        files = {"file": (file_name, buffer, mime_type)}
        return data, files

    @staticmethod
    def _build_transcription_usage(response_payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "prompt_tokens": None,
            "completion_tokens": None,
            "audio_seconds": response_payload.get("duration"),
        }

    def _transcribe_audio(
        self,
        audio_file,
        additional_parameters: AdditionalParameters | None = None,
    ) -> Dict[str, Any]:
        additional_parameters = additional_parameters or {}
        data, files = self._build_transcription_request(audio_file, additional_parameters)
        file_buffer = files["file"][1]

        try:
            response = requests.post(
                GROK_STT_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()
        finally:
            file_buffer.close()

    def _transcribe_conversation_audio(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters | None = None,
    ) -> Message:
        audio_file = self._extract_audio_file_for_transcription(the_conversation)
        if audio_file is None:
            return self._error_message(
                the_conversation,
                "Grok transcription models require exactly one audio file in the last user message.",
            )

        if len(audio_file.file_bytes) > GROK_STT_MAX_FILE_SIZE_BYTES:
            return self._error_message(
                the_conversation,
                "Grok transcription uploads are limited to 500 MB per file.",
            )

        try:
            response_payload = self._transcribe_audio(
                audio_file=audio_file,
                additional_parameters=additional_parameters,
            )
        except Exception as error:
            logger.error(f"Grok transcription failed: {error}")
            return self._error_message(the_conversation, f"Grok transcription failed: {error}")

        message = Message(
            role="assistant",
            content=self._format_transcription_response(
                response_payload=response_payload,
                additional_parameters=additional_parameters or {},
            ),
            usage=self._build_transcription_usage(response_payload, model),
        )
        the_conversation.messages.append(message)
        return message

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

        if self._is_transcription_model(model):
            if functions:
                logger.warning("Grok transcription models ignore function/tool definitions.")
            return self._transcribe_conversation_audio(
                model=model,
                the_conversation=the_conversation,
                additional_parameters=additional_parameters,
            )

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

    def voice_to_text(
        self,
        audio_file,
        audio_format: str | None = None,
        language: str | None = None,
        diarize: bool = False,
        **kwargs,
    ):
        additional_parameters = dict(kwargs)

        if audio_format:
            additional_parameters.setdefault("audio_format", audio_format)
        if language:
            additional_parameters.setdefault("language", language)
        if diarize is not None:
            additional_parameters.setdefault("diarize", diarize)

        response_payload = self._transcribe_audio(
            audio_file=audio_file,
            additional_parameters=additional_parameters,
        )
        return self._format_transcription_response(response_payload, additional_parameters)

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
