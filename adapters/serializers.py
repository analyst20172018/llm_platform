"""Provider wire (de)serialization for domain objects.

This module holds the conversion logic between the provider-agnostic domain
model (``llm_platform.services.conversation``) and the vendor-specific wire
formats used by the various provider APIs (OpenAI, Anthropic, Grok, ...).

It is deliberately kept in the adapters layer, out of the domain model, so that
``services/`` carries no vendor knowledge and stays provider-agnostic. Adapters
already depend on ``conversation``; ``conversation`` does NOT depend on
``adapters``, so importing the domain classes here introduces no import cycle.

Importing this module must not pull in any provider SDK.
"""

import json
from typing import Dict

from llm_platform.services.conversation import FunctionCall
from llm_platform.services.files import ImageFile, PDFDocumentFile, DocumentFile


def provider_dump(value):
    """Detach provider objects into lossless JSON data without importing SDKs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: provider_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [provider_dump(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json",
            exclude_unset=True,
            by_alias=True,
            serialize_as_any=True,
        )
    if hasattr(value, "DESCRIPTOR"):
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(value, preserving_proto_field_name=True)
    if hasattr(value, "__dict__"):
        return {key: provider_dump(item) for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError(f"Cannot preserve provider data of type {type(value).__name__}")


def missing_continuation(error, field: str, reference_id: str | None = None) -> bool:
    """Retry only an explicitly missing/expired continuation, never generic errors."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status not in (400, 404, 410):
        return False
    details = f"{error} {getattr(error, 'body', '')}".lower()
    identifies_state = field in details or (reference_id and reference_id.lower() in details)
    return bool(identifies_state and any(word in details for word in
            ("not found", "not_found", "expired", "deleted", "does not exist")))


def chat_replay_data(assistant_message):
    """Keep assistant input fields, including opaque reasoning and exact calls."""
    data = provider_dump(assistant_message)
    return {key: value for key, value in data.items() if key in {
        "role", "content", "tool_calls", "reasoning_content", "reasoning",
        "reasoning_details", "refusal", "audio", "annotations",
    }}


def openai_output_to_input(output):
    """Remove SDK parse conveniences that are not provider input fields."""
    items = provider_dump(output)
    for item in items:
        if item.get("type") == "function_call":
            item.pop("parsed_arguments", None)
        elif item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    content.pop("parsed", None)
    return items


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #

def function_call_from_openai(tool_call) -> FunctionCall:
    return FunctionCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=str(tool_call.arguments),
        call_id=getattr(tool_call, "call_id", tool_call.id),
        provider_data={"openai": {"caller": provider_dump(tool_call.caller)}}
        if getattr(tool_call, "caller", None) is not None else None,
    )


def function_call_to_openai(function_call) -> Dict:
    return {
        **function_call.provider_data.get("openai", {}),
        "id": function_call.id,
        "call_id": function_call.call_id,
        "name": function_call.name,
        "arguments": function_call.arguments if isinstance(function_call.arguments, str) else json.dumps(function_call.arguments),
        "type": "function_call",
    }


def function_response_to_openai(function_response) -> Dict:
    output = json.dumps(function_response.response)
    if function_response.files:
        output = [{"type": "input_text", "text": output}]
        for file in function_response.files:
            if isinstance(file, ImageFile):
                output.append({"type": "input_image", "image_url": f"data:{file.mime_type};base64,{file.base64}"})
            elif isinstance(file, PDFDocumentFile):
                output.append({"type": "input_file", "filename": file.name or "document.pdf",
                               "file_data": f"data:application/pdf;base64,{file.base64}"})
            elif isinstance(file, DocumentFile):
                output.append({"type": "input_text", "text": f"{file.name}\n{file.text}"})
            else:
                raise ValueError(f"Unsupported OpenAI tool attachment: {type(file).__name__}")
    return {
        **function_response.provider_data.get("openai", {}),
        "type": "function_call_output",
        "call_id": function_response.call_id,
        "output": output,
    }


def thinking_response_to_openai(thinking_response) -> Dict:
    return {
        "id": thinking_response.id,
        "summary": [
            {
                "type": "summary_text",
                "text": thinking_response.content,
            }
        ],
        "type": "reasoning",
    }


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions
# --------------------------------------------------------------------------- #
# The serializers above target the OpenAI *Responses* API. Providers on the
# OpenAI-compatible *Chat Completions* API (``chat.completions.create``) use a
# different tool-call wire shape: tool calls are nested under
# ``tool_calls[].function`` on the assistant message, and tool results are sent
# as standalone ``role: "tool"`` messages.

def function_call_from_openai_chat(tool_call) -> FunctionCall:
    return FunctionCall(
        id=tool_call.id,
        name=tool_call.function.name,
        arguments=str(tool_call.function.arguments),
        call_id=tool_call.id,
    )


def function_call_to_openai_chat(function_call) -> Dict:
    return {
        "id": function_call.call_id,
        "type": "function",
        "function": {
            "name": function_call.name,
            "arguments": function_call.arguments if isinstance(function_call.arguments, str) else json.dumps(function_call.arguments),
        },
    }


def function_response_to_openai_chat(function_response) -> Dict:
    function_response.require_no_files("Chat Completions")
    return {
        "role": "tool",
        "tool_call_id": function_response.call_id,
        "content": json.dumps(function_response.response),
    }


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

def function_call_to_anthropic(function_call) -> Dict:
    return {
        "id": function_call.call_id,
        "name": function_call.name,
        "input": json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments,
        "type": "tool_use",
    }


def function_response_to_anthropic(function_response) -> Dict:
    output = {
        "type": "tool_result",
        "tool_use_id": function_response.call_id,
        "content": [
            {
                "type": "text",
                "text": json.dumps(function_response.response),
            }
        ],
    }

    for file in function_response.files:
        if not isinstance(file, ImageFile):
            raise ValueError(f"Unsupported Anthropic tool attachment: {type(file).__name__}")
        output["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file.mime_type,
                    "data": file.base64,
                },
            }
        )

    return output


def thinking_response_to_anthropic(thinking_response) -> Dict:
    return {
        "type": "thinking",
        "thinking": thinking_response.content,
        "signature": thinking_response.id if thinking_response.id else "0",
    }


# --------------------------------------------------------------------------- #
# Grok
# --------------------------------------------------------------------------- #

def function_call_from_grok(tool_call) -> FunctionCall:
    return FunctionCall(
        id=tool_call.id,
        name=tool_call.function.name,
        arguments=str(tool_call.function.arguments),
        call_id=tool_call.id,
    )
