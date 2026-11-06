"""Offline context-budget regressions for both Anthropic request paths."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_platform.adapters.anthropic_adapter import (
    AnthropicAdapter,
    ContextWindowExceededError,
    TokenCountError,
)
from llm_platform.services.conversation import Conversation, Message


def make_adapter(input_tokens, asynchronous=False):
    adapter = AnthropicAdapter()
    adapter.model_config = {"test": NS(context_window=1_000_000, max_tokens=128_000)}
    client = MagicMock()
    count = (AsyncMock if asynchronous else MagicMock)(return_value=NS(input_tokens=input_tokens))
    create = (AsyncMock if asynchronous else MagicMock)()
    client.beta.messages.count_tokens = count
    client.beta.messages.create = create
    if asynchronous:
        adapter._async_client = client
    else:
        adapter._client = client
    adapter._round_kwargs = MagicMock(return_value={
        "model": "test", "messages": [{"role": "user", "content": "news"}],
        "system": "Analyze the evidence.", "tools": [], "max_tokens": 128_000,
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
        "cache_control": {"type": "ephemeral"}, "betas": ["test-beta"],
    })
    adapter._parse_non_streaming_response = MagicMock(return_value=NS())
    adapter._message_from_processor = MagicMock(return_value=Message("assistant", "done", status="completed"))
    return adapter, count, create


def request(adapter, asynchronous):
    state = Conversation([Message("user", "news")], "Analyze the evidence.")
    if asynchronous:
        return asyncio.run(adapter.request_llm_async("test", state, additional_parameters={"max_tokens": 128_000}))
    return adapter.request_llm("test", state, additional_parameters={"max_tokens": 128_000})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("input_tokens", [1_530_595, 1_000_000, 999_000])
def test_oversized_prompt_never_reaches_generation(asynchronous, input_tokens):
    adapter, count, create = make_adapter(input_tokens, asynchronous)
    with pytest.raises(ContextWindowExceededError, match="Reduce or split") as error:
        request(adapter, asynchronous)
    assert error.value.input_tokens == input_tokens
    assert error.value.context_window == 1_000_000
    assert error.value.max_tokens == 128_000
    count.assert_called_once()
    create.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_count_failure_stops_generation_and_preserves_cause(asynchronous):
    adapter, count, create = make_adapter(0, asynchronous)
    cause = RuntimeError("count service unavailable")
    count.side_effect = cause
    with pytest.raises(TokenCountError) as error:
        request(adapter, asynchronous)
    assert error.value.__cause__ is cause
    create.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_count_uses_full_request_and_stream_choice_uses_corrected_limit(asynchronous):
    adapter, count, create = make_adapter(989_000, asynchronous)
    result = request(adapter, asynchronous)
    assert result.content == "done"
    counted = count.call_args.kwargs
    sent = create.call_args.kwargs
    for name in ("model", "system", "messages", "tools", "thinking", "output_config", "betas", "cache_control"):
        assert counted[name] == sent[name]
    assert sent["max_tokens"] == 10_000
    assert not sent.get("stream", False)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fixed_thinking_budget_requires_room_for_visible_output(asynchronous):
    adapter, count, create = make_adapter(990_000, asynchronous)
    adapter._round_kwargs.return_value["thinking"] = {"type": "enabled", "budget_tokens": 16_000}
    with pytest.raises(ContextWindowExceededError):
        request(adapter, asynchronous)
    create.assert_not_called()


@pytest.mark.parametrize("value", [-1, "100", None, True])
def test_invalid_provider_token_count_is_not_treated_as_available_space(value):
    adapter, count, create = make_adapter(value)
    with pytest.raises(TokenCountError):
        request(adapter, False)
    create.assert_not_called()


def test_valid_limits_and_buffer_are_preserved():
    adapter, _, _ = make_adapter(0)
    assert adapter._clamp_max_tokens("test", 100, 4_096) == 4_096
    assert adapter._clamp_max_tokens("test", 100, None) == 128_000
    assert adapter._clamp_max_tokens("test", 100, 200_000) == 128_000
    assert adapter._clamp_max_tokens("test", 998_999, 100) == 1
    with pytest.raises(ValueError, match="max_tokens"):
        adapter._clamp_max_tokens("test", 100, 0)
    adapter.model_config["test"].max_tokens = 8_192
    with pytest.raises(ValueError, match="Model output limit"):
        adapter._clamp_max_tokens("test", 100, 128_000, minimum_output_tokens=16_001)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_serializes_matching_count_and_generation_payloads(asynchronous):
    import anthropic
    import httpx2

    from llm_platform.helpers.model_config import Model

    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if request.url.path.endswith("count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 120})
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "test",
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 120, "output_tokens": 3},
        })

    adapter = AnthropicAdapter()
    adapter.model_config = {"test": Model({
        "name": "test", "context_window": 1_000_000, "adaptive_thinking": True,
        "additional_parameters": {"max_tokens": {"default": 128_000}},
    })}
    state = Conversation([Message("user", "Evidence")], "Preserve all source dates.")
    parameters = {"max_tokens": 4_096, "reasoning": {"effort": "high"}}
    if asynchronous:
        async def run():
            adapter._async_client = anthropic.AsyncAnthropic(
                api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
            )
            try:
                return await adapter.request_llm_async("test", state, additional_parameters=parameters)
            finally:
                await adapter.async_client.close()
        result = asyncio.run(run())
    else:
        adapter._client = anthropic.Anthropic(
            api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        )
        try:
            result = adapter.request_llm("test", state, additional_parameters=parameters)
        finally:
            adapter.client.close()
    assert result.content == "done"
    assert len(payloads) == 2
    counted, generated = payloads
    for name in ("model", "system", "messages", "tools", "thinking", "output_config", "cache_control"):
        assert counted[name] == generated[name]
