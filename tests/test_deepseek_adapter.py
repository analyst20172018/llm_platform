import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from llm_platform.adapters.deepseek_adapter import DeepSeekAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message


MODEL = "deepseek-v4-pro"


def make_response(content: str = "Answer.", *, reasoning_content: str | None = None):
    assistant_message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(
        id="chat_1",
        choices=[SimpleNamespace(message=assistant_message)],
        usage=usage,
    )


def make_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role="user", content="Hello")],
        system_prompt="Be helpful.",
    )


def test_reasoning_defaults_are_normalized_from_the_model_config():
    handler = APIHandler()

    assert isinstance(handler.get_adapter(MODEL), DeepSeekAdapter)
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "max_tokens": 384_000,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert handler._prepare_additional_parameters(
        MODEL, {"thinking_mode": "disabled", "reasoning_effort": "low"}
    ) == {
        "max_tokens": 384_000,
        "thinking": {"type": "disabled"},
        "reasoning_effort": "low",
    }


def test_thinking_mode_travels_in_extra_body_and_drops_sampling_parameters():
    adapter = DeepSeekAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = make_response(
        reasoning_content="9.11 is smaller than 9.8."
    )
    conversation = make_conversation()

    message = adapter.request_llm(
        MODEL,
        conversation,
        additional_parameters={
            "max_tokens": 123,
            "temperature": 0.2,
            "top_p": 0.5,
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"},
        },
    )

    request = adapter._client.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["reasoning_effort"] == "max"
    assert request["max_tokens"] == 123
    assert "thinking" not in request
    assert "temperature" not in request
    assert "top_p" not in request
    assert message.thinking_responses[0].content == "9.11 is smaller than 9.8."
    assert message.usage == {
        "model": MODEL,
        "completion_tokens": 5,
        "prompt_tokens": 10,
    }
    assert conversation.messages[-1] is message


def test_sampling_parameters_survive_when_thinking_is_disabled():
    adapter = DeepSeekAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = make_response()

    message = adapter.request_llm(
        MODEL,
        make_conversation(),
        additional_parameters={
            "temperature": 0.2,
            "thinking": {"type": "disabled"},
        },
    )

    request = adapter._client.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["temperature"] == 0.2
    assert message.thinking_responses == []


def test_async_request_captures_reasoning():
    adapter = DeepSeekAdapter()
    adapter._async_client = MagicMock()
    adapter._async_client.chat.completions.create = AsyncMock(
        return_value=make_response(reasoning_content="Let me think.")
    )

    message = asyncio.run(
        adapter.request_llm_async(
            MODEL,
            make_conversation(),
            additional_parameters={"thinking": {"type": "enabled"}},
        )
    )

    request = adapter._async_client.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert message.thinking_responses[0].content == "Let me think."
