import base64
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

from llm_platform.services.files import (
    AudioFile,
    BaseFile,
    ByteDocumentFile,
    ExcelDocumentFile,
    ImageFile,
    MediaFile,
    PDFDocumentFile,
    TextDocumentFile,
    VideoFile,
    WordDocumentFile,
    PowerPointDocumentFile,
)

class FunctionCall:
    def __init__(
        self,
        id: str,
        name: str,
        arguments: str | Dict | List[Dict],
        call_id: str = None,
        provider_data: Dict | None = None,
    ):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.call_id = id if call_id is None else call_id
        self.provider_data = deepcopy(provider_data) if provider_data is not None else {}

    def __str__(self):
        return f"Id: {self.id}; Function: {self.name}, Arguments: {self.arguments}"

class FunctionResponse:
    def __init__(
        self,
        name: str,
        response: Dict,
        id: str = None,
        call_id: str = None,
        provider_data: Dict | None = None,
    ):
        self.name = name
        self.id = id
        self.call_id = id if call_id is None else call_id
        self.provider_data = deepcopy(provider_data) if provider_data is not None else {}
        self.response = deepcopy(response) if isinstance(response, dict) else {"text": response}
        self.files = []

        self._parse_response()

    def _parse_response(self):
        response_files = self.response.pop("files", None)
        if response_files is None:
            return

        assert isinstance(response_files, list), "`files` must be a list"

        for file in response_files:
            assert "type" in file, "File must have a 'type' key"
            if file["type"] != "image":
                continue

            assert "source" in file, "File must have a 'source' key"
            source = file["source"]
            assert "type" in source, "File source must have a 'type' key"
            assert "format" in source, "File source must have a 'format' key"

            if source["type"] == "base64":
                assert "data" in source, "File source must have a 'type' data"
                self.files.append(
                    ImageFile.from_base64(
                        base64_str=source["data"],
                        file_name=f"image.{source['format']}",
                    )
                )

    def __str__(self):
        return (
            f"Id: {self.id}; Call id: {self.call_id}; Function: {self.name}, "
            f"Response: {json.dumps(self.response)}"
        )

class ThinkingResponse:
    def __init__(self, content: str, id: str = None):
        self.content = content
        self.id = id

    def __str__(self):
        return f"Id: {self.id}; Thinking: {self.content}"

class Message:
    def __init__(
        self,
        role: str,
        content: str,
        thinking_responses: List[ThinkingResponse] | None = None,
        usage: Dict | None = None,
        files: List[BaseFile] | None = None,
        function_calls: List[FunctionCall] | None = None,
        function_responses: List[FunctionResponse] | None = None,
        id=None,
        additional_responses: List[Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        provider_data: Dict | None = None,
        status: str = "unknown",
        finish_reason: str | None = None,
        error: Any = None,
        incomplete_details: Any = None,
        citations: List[Dict] | None = None,
        hosted_tool_results: List[Dict] | None = None,
    ):
        assert role in ["user", "assistant", "function"]

        self.role = role
        self.content = content
        self.thinking_responses = [] if thinking_responses is None else thinking_responses
        self.timestamp = datetime.now()
        self.files = [] if files is None else files
        self.usage = usage
        self.function_calls = [] if function_calls is None else function_calls
        self.function_responses = [] if function_responses is None else function_responses
        self.id = id
        self.additional_responses = [] if additional_responses is None else additional_responses
        self.provider = provider
        self.model = model
        self.provider_data = deepcopy(provider_data) if provider_data is not None else {}
        self.status = status
        self.finish_reason = finish_reason
        self.error = deepcopy(error)
        self.incomplete_details = deepcopy(incomplete_details)
        self.citations = deepcopy(citations) if citations is not None else []
        self.hosted_tool_results = deepcopy(hosted_tool_results) if hosted_tool_results is not None else []
        self._replay_fingerprint = self._content_fingerprint()

    def _content_fingerprint(self):
        data = Conversation._serialize_message(self)
        for key in ("provider_data", "replay_fingerprint", "timestamp", "usage", "function_responses"):
            data.pop(key, None)
        for key in ("status", "finish_reason", "error", "incomplete_details", "citations", "hosted_tool_results"):
            data.pop(key, None)
        return Conversation._fingerprint(data)

    def replay_data(self, provider: str, model: str | None = None) -> Dict:
        """Return native replay data only for its origin and unchanged content."""
        if self.provider != provider or (model is not None and self.model != model):
            return {}
        if self._replay_fingerprint != self._content_fingerprint():
            return {}
        return deepcopy(self.provider_data)

    @property
    def can_execute_tools(self):
        """Only a response requesting client action may execute local tools."""
        return self.status == "requires_action"

    @property
    def text(self):
        return self.content

    def __str__(self):
        return (
            f"{self.role}: {self.content};"
            + "\n".join(str(thinking_response) for thinking_response in self.thinking_responses)
            + "\n"
            + "\n".join(str(function_call) for function_call in self.function_calls)
            + "\n\n"
            + "\n".join(str(function_response) for function_response in self.function_responses)
            + "\n".join(str(additional_response) for additional_response in self.additional_responses)
        )

class Conversation:
    def __init__(
        self,
        messages: List[Message] | None = None,
        system_prompt: str | None = None,
        continuations: Dict | None = None,
    ):
        self.messages = list(messages) if messages is not None else []
        self.system_prompt = system_prompt
        self.continuations = deepcopy(continuations) if continuations is not None else {}

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)

    def clear(self):
        self.messages.clear()
        self.continuations.clear()

    @staticmethod
    def _fingerprint(data) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _prefix_fingerprint(self, length: int) -> str:
        return self._fingerprint({
            "system_prompt": self.system_prompt,
            "messages": [self._serialize_message(m) for m in self.messages[:length]],
        })

    @staticmethod
    def _continuation_key(provider: str, model: str) -> str:
        return json.dumps([provider, model])

    def checkpoint(self, provider: str, model: str, response_id: str, **state):
        """Record exactly the local prefix already represented by remote state."""
        if response_id:
            self.continuations[self._continuation_key(provider, model)] = {
                **state, "response_id": response_id, "length": len(self.messages),
                "fingerprint": self._prefix_fingerprint(len(self.messages)),
            }

    def continuation(self, provider: str, model: str) -> Dict | None:
        state = self.continuations.get(self._continuation_key(provider, model))
        if not state:
            return None
        length = state.get("length", -1)
        if not 0 <= length <= len(self.messages):
            return None
        if state.get("fingerprint") != self._prefix_fingerprint(length):
            return None
        return deepcopy(state)

    def reset_continuation(self, provider: str, model: str):
        """Start fresh remote state (including a new managed-agent environment)."""
        self.continuations.pop(self._continuation_key(provider, model), None)

    @staticmethod
    def _empty_usage() -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "costs": 0,
        }

    @staticmethod
    def _usage_value(message: Message, key: str) -> Any:
        if not message.usage:
            return 0

        value = message.usage.get(key, 0)
        return 0 if value is None else value

    @property
    def usage_total(self) -> Dict:
        total_usage = self._empty_usage()
        for message in self.messages:
            total_usage["prompt_tokens"] += self._usage_value(message, "prompt_tokens")
            total_usage["completion_tokens"] += self._usage_value(message, "completion_tokens")
            total_usage["costs"] += self._usage_value(message, "costs")
            for key in ("cache_read_tokens", "cache_creation_tokens", "reasoning_tokens"):
                if message.usage and message.usage.get(key) is not None:
                    total_usage[key] = total_usage.get(key, 0) + self._usage_value(message, key)
        return total_usage

    @property
    def usage_last(self) -> Dict:
        if self.messages and self.messages[-1].usage:
            return self.messages[-1].usage
        return self._empty_usage()

    @property
    def last_assistant_id(self):
        """Legacy display helper. Never use this unscoped ID for continuation."""
        for message in reversed(self.messages):
            if message.role == "assistant" and message.id:
                return message.id
        return None

    def save_to_json(self) -> Dict:
        return deepcopy({
            "version": 2,
            "system_prompt": self.system_prompt,
            "continuations": deepcopy(self.continuations),
            "messages": [self._serialize_message(message) for message in self.messages],
        })

    @classmethod
    def _serialize_message(cls, message: Message) -> Dict:
        return {
            "status": message.status,
            "finish_reason": message.finish_reason,
            "error": deepcopy(message.error),
            "incomplete_details": deepcopy(message.incomplete_details),
            "citations": deepcopy(message.citations),
            "hosted_tool_results": deepcopy(message.hosted_tool_results),
            "id": message.id,
            "provider": message.provider,
            "model": message.model,
            "provider_data": deepcopy(message.provider_data),
            "replay_fingerprint": getattr(message, "_replay_fingerprint", None),
            "additional_responses": deepcopy(message.additional_responses),
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "usage": message.usage,
            "thinking_responses": [
                {"content": thinking_response.content, "id": thinking_response.id}
                for thinking_response in message.thinking_responses
            ],
            "function_calls": [
                {
                    "id": function_call.id,
                    "call_id": function_call.call_id,
                    "provider_data": deepcopy(function_call.provider_data),
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                }
                for function_call in message.function_calls
            ],
            "function_responses": [
                {
                    "name": function_response.name,
                    "id": function_response.id,
                    "call_id": function_response.call_id,
                    "provider_data": deepcopy(function_response.provider_data),
                    "response": deepcopy(function_response.response),
                    "files": cls._serialize_files(function_response.files),
                }
                for function_response in message.function_responses
            ],
            "files": cls._serialize_files(message.files),
        }

    @classmethod
    def _serialize_files(cls, files: List[BaseFile]) -> List[Dict]:
        return [cls._serialize_file(file) for file in files]

    @staticmethod
    def _serialize_file(file: BaseFile) -> Dict[str, Any]:
        file_data = {
            "name": file.name,
            "type": type(file).__name__,
        }

        if isinstance(file, TextDocumentFile):
            file_data["text"] = file.text
        elif isinstance(file, ByteDocumentFile):
            file_data["base64"] = file.base64
        elif isinstance(file, MediaFile):
            file_data["base64"] = file.base64
            file_data["extension"] = file.extension

        return file_data

    @classmethod
    def read_from_json(cls, data: Dict) -> "Conversation":
        if data.get("version", 1) not in (1, 2):
            raise ValueError("Unsupported conversation persistence version")
        messages = [cls._deserialize_message(message_data) for message_data in data.get("messages", [])]
        return cls(messages=messages, system_prompt=data.get("system_prompt"),
                   continuations=data.get("continuations"))

    @classmethod
    def _deserialize_message(cls, message_data: Dict) -> Message:
        message_data = deepcopy(message_data)
        message = Message(
            status=message_data.get("status", "unknown"),
            finish_reason=message_data.get("finish_reason"),
            error=message_data.get("error"),
            incomplete_details=message_data.get("incomplete_details"),
            citations=message_data.get("citations"),
            hosted_tool_results=message_data.get("hosted_tool_results"),
            id=message_data.get("id"),
            provider=message_data.get("provider"),
            model=message_data.get("model"),
            provider_data=message_data.get("provider_data"),
            additional_responses=deepcopy(message_data.get("additional_responses", [])),
            role=message_data["role"],
            content=message_data["content"],
            thinking_responses=[
                ThinkingResponse(content=thinking_data["content"], id=thinking_data.get("id"))
                for thinking_data in message_data.get("thinking_responses", [])
            ],
            usage=message_data.get("usage"),
            files=cls._deserialize_files(message_data.get("files", [])),
            function_calls=[
                FunctionCall(
                    id=function_call["id"],
                    name=function_call["name"],
                    arguments=function_call["arguments"],
                    call_id=function_call.get("call_id"),
                    provider_data=function_call.get("provider_data"),
                )
                for function_call in message_data.get("function_calls", [])
            ],
            function_responses=cls._deserialize_function_responses(
                message_data.get("function_responses", [])
            ),
        )

        if "timestamp" in message_data:
            message.timestamp = datetime.fromisoformat(message_data["timestamp"])
        if message_data.get("replay_fingerprint"):
            message._replay_fingerprint = message_data["replay_fingerprint"]

        return message

    @classmethod
    def _deserialize_function_responses(
        cls,
        function_response_data: List[Dict],
    ) -> List[FunctionResponse]:
        function_responses = []
        for response_data in function_response_data:
            function_response = FunctionResponse(
                name=response_data["name"],
                response=response_data["response"],
                id=response_data.get("id"),
                call_id=response_data.get("call_id"),
                provider_data=response_data.get("provider_data"),
            )
            function_response.files = cls._deserialize_files(response_data.get("files", []))
            function_responses.append(function_response)
        return function_responses

    @classmethod
    def _deserialize_files(cls, file_data_list: List[Dict]) -> List[BaseFile]:
        files = []
        for file_data in file_data_list:
            file = cls._deserialize_file(file_data)
            if file is not None:
                files.append(file)
        return files

    @staticmethod
    def _decode_base64_file(file_data: Dict) -> bytes:
        return base64.b64decode(file_data["base64"])

    @classmethod
    def _deserialize_file(cls, file_data: Dict) -> BaseFile | None:
        if "type" not in file_data or "name" not in file_data:
            return None

        file_type = file_data["type"]
        file_name = file_data["name"]

        if file_type == "TextDocumentFile" and "text" in file_data:
            return TextDocumentFile(text=file_data["text"], name=file_name)
        if file_type == "ImageFile" and "base64" in file_data:
            return ImageFile.from_base64(file_data["base64"], file_name)
        if file_type == "PDFDocumentFile" and "base64" in file_data:
            return PDFDocumentFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "AudioFile" and "base64" in file_data:
            # Persisted audio already contains converted bytes, even when the
            # original name has another extension. Do not transcode it again.
            audio = AudioFile.__new__(AudioFile)
            MediaFile.__init__(audio, cls._decode_base64_file(file_data), file_name)
            return audio
        if file_type == "ExcelDocumentFile" and "base64" in file_data:
            return ExcelDocumentFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "VideoFile" and "base64" in file_data:
            return VideoFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        document_types = {c.__name__: c for c in (ByteDocumentFile, WordDocumentFile, PowerPointDocumentFile)}
        if file_type in document_types and "base64" in file_data:
            return document_types[file_type].from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "MediaFile" and "base64" in file_data:
            return MediaFile.from_bytes(cls._decode_base64_file(file_data), file_name)

        return None
