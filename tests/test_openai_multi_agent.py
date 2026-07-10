"""Tests for OpenAI Multi-agent (beta) support in OpenAIAdapter.

`agent_count` > 0 enables hosted subagent orchestration through the beta
Responses client, the unsupported `reasoning.summary` is not requested, and
response parsing keeps only one copy of the root agent's final answer as
user-facing text.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.services.conversation import Conversation, Message

MODEL = "gpt-test"
BETA_FEATURES = ["responses_multi_agent=v1"]


class Answer(BaseModel):
    value: int


def make_adapter() -> OpenAIAdapter:
    """Adapter with a mocked client: no API key or network access needed."""
    fake_response = SimpleNamespace(id="resp_1", model=MODEL, usage=None, output=[])

    adapter = OpenAIAdapter()
    adapter.model_config = {MODEL: {"background_mode": False}}

    adapter._client = MagicMock()
    adapter._client.responses.create.return_value = fake_response
    adapter._client.beta.responses.create.return_value = fake_response

    adapter._async_client = MagicMock()
    adapter._async_client.responses.create = AsyncMock(return_value=fake_response)
    adapter._async_client.beta.responses.create = AsyncMock(return_value=fake_response)
    return adapter


def make_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role="user", content="Compare alpha and beta.")],
        system_prompt="Be brief.",
    )


def message_item(text: str, agent_name: str = None, phase: str = None) -> SimpleNamespace:
    content = [SimpleNamespace(type="output_text", text=text, annotations=[])]
    item = SimpleNamespace(type="message", content=content)
    if agent_name is not None:
        item.agent = SimpleNamespace(agent_name=agent_name)
        item.phase = phase
    return item


def test_agent_count_enables_multi_agent_payload():
    adapter = make_adapter()
    adapter.request_llm(MODEL, make_conversation(), additional_parameters={"agent_count": 3})

    kwargs = adapter._client.beta.responses.create.call_args.kwargs
    assert kwargs["multi_agent"] == {
        "enabled": True,
        "max_concurrent_subagents": 3,
    }
    assert kwargs["betas"] == BETA_FEATURES
    assert "agent_count" not in kwargs
    adapter._client.responses.create.assert_not_called()


def test_agent_count_accepts_string_values():
    """Enum values arrive from the normalizer as strings."""
    adapter = make_adapter()
    adapter.request_llm(MODEL, make_conversation(), additional_parameters={"agent_count": "5"})

    kwargs = adapter._client.beta.responses.create.call_args.kwargs
    assert kwargs["multi_agent"]["max_concurrent_subagents"] == 5


@pytest.mark.parametrize("agent_count", [None, 0, "0"])
def test_multi_agent_off_sends_standard_request(agent_count):
    adapter = make_adapter()
    additional_parameters = {} if agent_count is None else {"agent_count": agent_count}
    adapter.request_llm(MODEL, make_conversation(), additional_parameters=additional_parameters)

    kwargs = adapter._client.responses.create.call_args.kwargs
    assert "multi_agent" not in kwargs
    assert "betas" not in kwargs
    adapter._client.beta.responses.create.assert_not_called()


def test_multi_agent_async_uses_beta_responses_client():
    adapter = make_adapter()
    asyncio.run(
        adapter.request_llm_async(
            MODEL,
            make_conversation(),
            additional_parameters={"agent_count": 2},
        )
    )

    adapter._async_client.beta.responses.create.assert_awaited_once()
    adapter._async_client.responses.create.assert_not_awaited()


def test_reasoning_summary_suppressed_with_multi_agent():
    adapter = make_adapter()
    adapter.request_llm(
        MODEL,
        make_conversation(),
        additional_parameters={"agent_count": 3, "reasoning": {"effort": "high"}},
    )

    kwargs = adapter._client.beta.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}


def test_reasoning_summary_kept_without_multi_agent():
    adapter = make_adapter()
    adapter.request_llm(
        MODEL, make_conversation(), additional_parameters={"reasoning": {"effort": "high"}}
    )

    kwargs = adapter._client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}


def test_structured_output_with_multi_agent_raises():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="multi-agent"):
        adapter.request_llm(
            MODEL,
            make_conversation(),
            additional_parameters={"agent_count": 3, "structured_output": Answer},
        )


def test_parse_response_keeps_only_root_final_answer():
    adapter = make_adapter()
    response = SimpleNamespace(
        id="resp_1",
        model=MODEL,
        usage=None,
        output=[
            message_item("subagent findings", agent_name="/root/researcher", phase="final_answer"),
            message_item("root planning note", agent_name="/root", phase="planning"),
            message_item("Final answer.", agent_name="/root", phase="final_answer"),
            SimpleNamespace(
                type="multi_agent_call",
                agent=SimpleNamespace(agent_name="/root"),
                phase=None,
            ),
        ],
    )

    answer_text, _, _, _, _ = adapter._parse_response(response)
    assert answer_text == "Final answer."


def test_parse_response_deduplicates_repeated_root_final_answer():
    adapter = make_adapter()
    response = SimpleNamespace(
        id="resp_1",
        model=MODEL,
        usage=None,
        output=[
            message_item("Final answer.", agent_name="/root", phase="final_answer"),
            message_item("Final answer.", agent_name="/root", phase="final_answer"),
        ],
    )

    answer_text, _, _, _, _ = adapter._parse_response(response)
    assert answer_text == "Final answer."


def test_parse_response_keeps_distinct_root_final_fragments():
    adapter = make_adapter()
    response = SimpleNamespace(
        id="resp_1",
        model=MODEL,
        usage=None,
        output=[
            message_item("Part one.", agent_name="/root", phase="final_answer"),
            message_item("Part two.", agent_name="/root", phase="final_answer"),
        ],
    )

    answer_text, _, _, _, _ = adapter._parse_response(response)
    assert answer_text == "Part one.\nPart two."


def test_parse_response_unchanged_for_standard_responses():
    adapter = make_adapter()
    response = SimpleNamespace(
        id="resp_1",
        model=MODEL,
        usage=None,
        output=[message_item("Plain answer.")],
    )

    answer_text, _, _, _, _ = adapter._parse_response(response)
    assert answer_text == "Plain answer."


def test_function_calls_extracted_from_subagents():
    adapter = make_adapter()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                name="get_proposal",
                arguments='{"proposal": "alpha"}',
                call_id="call_1",
                agent=SimpleNamespace(agent_name="/root/researcher"),
            ),
        ],
    )

    function_calls = adapter._get_function_calls_from_response(response)
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_proposal"
