"""Regression tests for structured-output routing in OpenAIAdapter.

In the OpenAI SDK, `text_format` is accepted only by `responses.parse()`,
not `responses.create()`, so every request path (sync/async, with/without
tools) must branch to `parse` when structured output is requested.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.services.conversation import Conversation, Message

MODEL = "gpt-test"


class Answer(BaseModel):
    value: int


def make_adapter() -> OpenAIAdapter:
    """Adapter with mocked clients: no API key or network access needed."""
    fake_response = SimpleNamespace(id="resp_1", model=MODEL, usage=None, output=[])

    adapter = OpenAIAdapter()
    adapter.model_config = {MODEL: {"background_mode": False}}

    adapter._client = MagicMock()
    adapter._client.responses.parse.return_value = fake_response
    adapter._client.responses.create.return_value = fake_response

    adapter._async_client = MagicMock()
    adapter._async_client.responses.parse = AsyncMock(return_value=fake_response)
    adapter._async_client.responses.create = AsyncMock(return_value=fake_response)
    return adapter


def make_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role="user", content="value=1?")],
        system_prompt="Return JSON.",
    )


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return "sunny"


def test_request_llm_uses_parse_for_structured_output():
    adapter = make_adapter()
    adapter.request_llm(
        MODEL, make_conversation(), additional_parameters={"structured_output": Answer}
    )

    adapter._client.responses.parse.assert_called_once()
    adapter._client.responses.create.assert_not_called()
    assert adapter._client.responses.parse.call_args.kwargs["text_format"] is Answer


def test_request_llm_with_functions_uses_parse_for_structured_output():
    adapter = make_adapter()
    adapter.request_llm(
        MODEL,
        make_conversation(),
        functions=[get_weather],
        additional_parameters={"structured_output": Answer},
    )

    adapter._client.responses.parse.assert_called_once()
    adapter._client.responses.create.assert_not_called()


def test_request_llm_async_uses_parse_for_structured_output():
    adapter = make_adapter()
    asyncio.run(
        adapter.request_llm_async(
            MODEL, make_conversation(), additional_parameters={"structured_output": Answer}
        )
    )

    adapter._async_client.responses.parse.assert_awaited_once()
    adapter._async_client.responses.create.assert_not_awaited()
    assert adapter._async_client.responses.parse.call_args.kwargs["text_format"] is Answer


def test_request_llm_async_uses_create_without_structured_output():
    adapter = make_adapter()
    asyncio.run(adapter.request_llm_async(MODEL, make_conversation()))

    adapter._async_client.responses.create.assert_awaited_once()
    adapter._async_client.responses.parse.assert_not_awaited()


def test_request_llm_with_functions_async_uses_parse_for_structured_output():
    adapter = make_adapter()
    asyncio.run(
        adapter.request_llm_async(
            MODEL,
            make_conversation(),
            functions=[get_weather],
            additional_parameters={"structured_output": Answer},
        )
    )

    adapter._async_client.responses.parse.assert_awaited_once()
    adapter._async_client.responses.create.assert_not_awaited()
