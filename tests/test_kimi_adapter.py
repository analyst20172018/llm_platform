import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from llm_platform.adapters.kimi_adapter import KimiAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message, ThinkingResponse
from llm_platform.services.files import ImageFile, VideoFile


MODEL = "kimi-k3"


class Answer(BaseModel):
    value: int


def make_response(
    content: str = "",
    *,
    reasoning_content: str | None = None,
    tool_calls=None,
    response_id: str = "chat_1",
):
    assistant_message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(
        id=response_id,
        choices=[SimpleNamespace(message=assistant_message)],
        usage=usage,
    )


def make_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role="user", content="Hello")],
        system_prompt="Be helpful.",
    )


def test_kimi_model_is_registered_with_lazy_adapter():
    handler = APIHandler()

    adapter = handler.get_adapter(MODEL)

    assert isinstance(adapter, KimiAdapter)
    assert adapter._client is None
    assert handler.model_config[MODEL].context_window == 1_048_576
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "max_completion_tokens": 131072
    }


def test_kimi_client_requires_moonshot_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
        KimiAdapter()._build_client()


def test_request_maps_kimi_parameters_and_captures_reasoning():
    adapter = KimiAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = make_response(
        '{"value": 1}', reasoning_content="I should extract the value."
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
            "structured_output": Answer,
        },
    )

    request = adapter._client.chat.completions.create.call_args.kwargs
    assert request["max_completion_tokens"] == 123
    assert request["reasoning_effort"] == "max"
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert "top_p" not in request
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "strict": True,
            "schema": Answer.model_json_schema(),
        },
    }
    assert message.thinking_responses[0].content == "I should extract the value."
    assert message.usage == {
        "model": MODEL,
        "completion_tokens": 5,
        "prompt_tokens": 10,
    }


def test_conversation_preserves_reasoning_and_multimodal_parts():
    conversation = Conversation(
        messages=[
            Message(
                role="user",
                content="Compare these.",
                files=[
                    ImageFile.from_bytes(b"image", "sample.png"),
                    VideoFile.from_bytes(b"video", "sample.mp4"),
                ],
            ),
            Message(
                role="assistant",
                content="Done.",
                thinking_responses=[ThinkingResponse(content="visual reasoning")],
            ),
        ],
        system_prompt="Be helpful.",
    )
    adapter = KimiAdapter()

    history, _ = adapter.convert_conversation_history_to_adapter_format(
        conversation, MODEL
    )

    user_parts = history[1]["content"]
    assert user_parts[1]["type"] == "image_url"
    assert user_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_parts[2]["type"] == "video_url"
    assert user_parts[2]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert history[2]["reasoning_content"] == "visual reasoning"


def test_tool_loop_returns_complete_assistant_history():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "Prague"}'),
    )
    adapter = KimiAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.side_effect = [
        make_response(reasoning_content="I need the weather.", tool_calls=[tool_call]),
        make_response("It is sunny.", reasoning_content="I have the result."),
    ]
    conversation = make_conversation()

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"sunny in {city}"

    message = adapter.request_llm(MODEL, conversation, functions=[get_weather])

    assert message.content == "It is sunny."
    assert adapter._client.chat.completions.create.call_count == 2
    second_request = adapter._client.chat.completions.create.call_args_list[1].kwargs
    assistant_history = second_request["messages"][2]
    tool_history = second_request["messages"][3]
    assert assistant_history["reasoning_content"] == "I need the weather."
    assert assistant_history["tool_calls"][0]["id"] == "call_1"
    assert tool_history == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"text": "sunny in Prague"}',
    }


def test_async_request_uses_native_async_client():
    adapter = KimiAdapter()
    adapter._async_client = MagicMock()
    adapter._async_client.chat.completions.create = AsyncMock(
        return_value=make_response("Async answer", reasoning_content="Async thought")
    )

    message = asyncio.run(adapter.request_llm_async(MODEL, make_conversation()))

    assert message.content == "Async answer"
    assert message.thinking_responses[0].content == "Async thought"
    adapter._async_client.chat.completions.create.assert_awaited_once()
