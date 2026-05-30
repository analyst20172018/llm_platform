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
from llm_platform.services.files import ImageFile


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #

def function_call_from_openai(tool_call) -> FunctionCall:
    return FunctionCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=str(tool_call.arguments),
        call_id=getattr(tool_call, "call_id", tool_call.id),
    )


def function_call_to_openai(function_call) -> Dict:
    return {
        "id": function_call.id,
        "call_id": function_call.call_id,
        "name": function_call.name,
        "arguments": function_call.arguments,
        "type": "function_call",
    }


def function_response_to_openai(function_response) -> Dict:
    if function_response.files:
        print("WARNING: Files are not supported in function responses for OpenAI")
    return {
        "type": "function_call_output",
        "call_id": function_response.call_id,
        "output": json.dumps(function_response.response),
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
# Anthropic
# --------------------------------------------------------------------------- #

def function_call_to_anthropic(function_call) -> Dict:
    return {
        "id": function_call.id,
        "name": function_call.name,
        "input": function_call.arguments,
        "type": "tool_use",
    }


def function_response_to_anthropic(function_response) -> Dict:
    output = {
        "type": "tool_result",
        "tool_use_id": function_response.id,
        "content": [
            {
                "type": "text",
                "text": json.dumps(function_response.response),
            }
        ],
    }

    for file in function_response.files:
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
