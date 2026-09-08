from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import BaseModel

from llm_platform.adapters.zai_adapter import ZaiAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message


MODEL = "glm-5.3-flash"


class Answer(BaseModel):
    value: int


def make_response(
    content: str = "",
    *,
    reasoning_content: str | None = None,
    tool_calls=None,
):
    return SimpleNamespace(
        id="chat_1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
            )
        ],
        usage=None,
    )


def make_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role="user", content="Return value 1.")],
        system_prompt="Be concise.",
    )


def test_glm_flash_reasoning_defaults_are_normalized_from_model_config():
    handler = APIHandler()

    assert isinstance(handler.get_adapter(MODEL), ZaiAdapter)
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "max_tokens": 128000,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "high",
        "web_search": False,
    }


def test_glm_flash_reasoning_effort_can_be_reduced():
    handler = APIHandler()

    parameters = handler._prepare_additional_parameters(
        MODEL,
        {"reasoning_effort": "low"},
    )

    assert parameters == {
        "max_tokens": 128000,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
        "web_search": False,
    }
    assert handler.get_adapter(MODEL)._build_request_params(MODEL, parameters) == {
        "max_tokens": 128000,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
    }


def test_glm_flash_structured_output_is_allowed_by_model_config():
    parameters = APIHandler()._prepare_additional_parameters(
        MODEL,
        {"structured_output": Answer},
    )

    assert parameters["structured_output"] is Answer


def test_structured_output_uses_json_mode_and_adds_schema_instruction():
    adapter = ZaiAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = make_response(
        '{"value": 1}',
        reasoning_content="I should return the requested object.",
    )

    message = adapter.request_llm(
        MODEL,
        make_conversation(),
        additional_parameters={"structured_output": Answer},
    )

    request = adapter._client.chat.completions.create.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}
    assert '"value"' in request["messages"][0]["content"]
    assert message.content == '{"value": 1}'
    assert message.thinking_responses[0].content == "I should return the requested object."


def test_function_calling_preserves_interleaved_thinking():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "Prague"}'),
    )
    adapter = ZaiAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.side_effect = [
        make_response(
            reasoning_content="I need the weather.",
            tool_calls=[tool_call],
        ),
        make_response(
            "It is sunny.",
            reasoning_content="I can now answer.",
        ),
    ]

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"sunny in {city}"

    message = adapter.request_llm(
        MODEL,
        make_conversation(),
        functions=[get_weather],
        additional_parameters={
            "thinking": {"type": "enabled", "clear_thinking": False},
            "structured_output": Answer,
        },
    )

    assert message.content == "It is sunny."
    assert message.thinking_responses[0].content == "I can now answer."

    second_request = adapter._client.chat.completions.create.call_args_list[1].kwargs
    assert second_request["response_format"] == {"type": "json_object"}
    assistant_history = second_request["messages"][2]
    tool_history = second_request["messages"][3]
    assert assistant_history["reasoning_content"] == "I need the weather."
    assert assistant_history["tool_calls"][0]["id"] == "call_1"
    assert tool_history == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"text": "sunny in Prague"}',
    }
