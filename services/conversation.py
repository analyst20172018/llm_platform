import base64
import json
from datetime import datetime
from typing import Any, Dict, List

from llm_platform.services.files import (
    AudioFile,
    BaseFile,
    ExcelDocumentFile,
    ImageFile,
    MediaFile,
    PDFDocumentFile,
    TextDocumentFile,
    VideoFile,
)


class FunctionCall:
    def __init__(
        self,
        id: str,
        name: str,
        arguments: Dict | List[Dict],
        call_id: str = None,
    ):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.call_id = id if call_id is None else call_id

    @classmethod
    def from_openai(cls, tool_call):
        return cls(
            id=tool_call.id,
            name=tool_call.name,
            arguments=str(tool_call.arguments),
            call_id=getattr(tool_call, "call_id", tool_call.id),
        )

    @classmethod
    def from_grok(cls, tool_call):
        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=str(tool_call.function.arguments),
            call_id=tool_call.id,
        )

    @classmethod
    def from_openai_old(cls, tool_call):
        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=str(tool_call.function.arguments),
            call_id=tool_call.id,
        )

    def to_openai(self) -> Dict:
        return {
            "id": self.id,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "type": "function_call",
        }

    def to_grok(self) -> Dict:
        return {
            "function": {
                "arguments": self.arguments,
                "name": self.name,
            },
            "id": self.call_id,
            "type": "function",
        }

    def to_openai_old(self) -> Dict:
        return {
            "id": self.id,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
            "type": "function",
        }

    def to_anthropic(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "input": self.arguments,
            "type": "tool_use",
        }

    def __str__(self):
        return f"Id: {self.id}; Function: {self.name}, Arguments: {self.arguments}"


class FunctionResponse:
    def __init__(
        self,
        name: str,
        response: Dict,
        id: str = None,
        call_id: str = None,
    ):
        self.name = name
        self.id = id
        self.call_id = id if call_id is None else call_id
        self.response = response if isinstance(response, dict) else {"text": response}
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

    def to_openai(self) -> Dict:
        if self.files:
            print("WARNING: Files are not supported in function responses for OpenAI")
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": json.dumps(self.response),
        }

    def to_openai_old(self) -> Dict:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": json.dumps(self.response),
        }

    def to_anthropic(self) -> Dict:
        output = {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(self.response),
                }
            ],
        }

        for file in self.files:
            if isinstance(file, ImageFile):
                output["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{file.extension}",
                            "data": file.base64,
                        },
                    }
                )

        return output

    def __str__(self):
        return (
            f"Id: {self.id}; Call id: {self.call_id}; Function: {self.name}, "
            f"Response: {json.dumps(self.response)}"
        )


class ThinkingResponse:
    def __init__(self, content: str, id: str = None):
        self.content = content
        self.id = id

    def to_openai(self) -> Dict:
        return {
            "id": self.id,
            "summary": [
                {
                    "type": "summary_text",
                    "text": self.content,
                }
            ],
            "type": "reasoning",
        }

    def to_anthropic(self) -> Dict:
        return {
            "type": "thinking",
            "thinking": self.content,
            "signature": self.id if self.id else "0",
        }

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
        additional_responses: List[str] | None = None,
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
    ):
        self.messages = list(messages) if messages is not None else []
        self.system_prompt = system_prompt

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)

    def clear(self):
        self.messages.clear()

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
        return total_usage

    @property
    def usage_last(self) -> Dict:
        if self.messages and self.messages[-1].usage:
            return self.messages[-1].usage
        return self._empty_usage()

    @property
    def previous_response_id_for_openai(self):
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.id
        return None

    def save_to_json(self) -> Dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": [self._serialize_message(message) for message in self.messages],
        }

    @classmethod
    def _serialize_message(cls, message: Message) -> Dict:
        return {
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
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                }
                for function_call in message.function_calls
            ],
            "function_responses": [
                {
                    "name": function_response.name,
                    "id": function_response.id,
                    "response": function_response.response,
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
        elif isinstance(file, PDFDocumentFile):
            file_data["base64"] = file.base64
            file_data["text"] = file.text
            file_data["number_of_pages"] = file.number_of_pages
        elif isinstance(file, ExcelDocumentFile):
            file_data["base64"] = file.base64
            file_data["text"] = file.text
        elif isinstance(file, VideoFile):
            if hasattr(file, "base64") and file.base64:
                file_data["base64"] = file.base64
            file_data["extension"] = file.extension
        elif isinstance(file, (ImageFile, AudioFile, MediaFile)):
            file_data["base64"] = file.base64
            file_data["extension"] = file.extension

        return file_data

    @classmethod
    def read_from_json(cls, data: Dict) -> "Conversation":
        messages = [cls._deserialize_message(message_data) for message_data in data.get("messages", [])]
        return cls(messages=messages, system_prompt=data.get("system_prompt"))

    @classmethod
    def _deserialize_message(cls, message_data: Dict) -> Message:
        message = Message(
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
                )
                for function_call in message_data.get("function_calls", [])
            ],
            function_responses=cls._deserialize_function_responses(
                message_data.get("function_responses", [])
            ),
        )

        if "timestamp" in message_data:
            message.timestamp = datetime.fromisoformat(message_data["timestamp"])

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
        if file_type == "ImageFile" and file_data.get("base64"):
            return ImageFile.from_base64(file_data["base64"], file_name)
        if file_type == "PDFDocumentFile" and file_data.get("base64"):
            return PDFDocumentFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "AudioFile" and file_data.get("base64"):
            return AudioFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "ExcelDocumentFile" and file_data.get("base64"):
            return ExcelDocumentFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "VideoFile" and file_data.get("base64"):
            return MediaFile.from_bytes(cls._decode_base64_file(file_data), file_name)
        if file_type == "MediaFile" and file_data.get("base64"):
            return MediaFile.from_bytes(cls._decode_base64_file(file_data), file_name)

        return None
