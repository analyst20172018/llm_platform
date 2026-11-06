"""Offline coverage for Pro reasoning and hosted parallel-agent compatibility."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_platform.adapters.adapter_base import AdapterBase
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.core.parameter_normalizer import ParameterNormalizer
from llm_platform.exceptions import RequestValidationError
from llm_platform.helpers.model_config import Model
from llm_platform.services.conversation import Message
from llm_platform.services.files import TextDocumentFile


MODEL = "gpt-6-astra"
REQUEST_METHODS = ("request", "request_async", "request_llm", "request_llm_async")
ERROR_MESSAGE = (
    "Pro reasoning mode cannot be combined with parallel agents. "
    "Select Standard mode to use parallel agents, or set agent_count "
    "to 0 to keep Pro mode."
)


def local_tool(value: str) -> str:
    return value


@pytest.fixture
def handler_and_adapter():
    handler = APIHandler(system_prompt="private system instructions")
    adapter = OpenAIAdapter()
    response = SimpleNamespace(
        id="resp_new", model=MODEL, usage=None, output=[], status="completed"
    )
    adapter._client = MagicMock()
    adapter._client.responses.create.return_value = response
    adapter._client.beta.responses.create.return_value = response
    adapter._async_client = MagicMock()
    adapter._async_client.responses.create = AsyncMock(return_value=response)
    adapter._async_client.beta.responses.create = AsyncMock(return_value=response)
    handler.adapters["OpenAIAdapter"] = adapter

    conversation = handler.the_conversation
    conversation.messages.extend([
        Message(
            role="user", content="private conversation history",
            files=[TextDocumentFile("private existing attachment", "history.txt")],
        ),
        Message(
            role="assistant", content="previous answer", id="resp_previous",
            provider="openai", model=MODEL,
            usage={"prompt_tokens": 11, "completion_tokens": 7, "costs": 0.25},
        ),
    ])
    conversation.checkpoint("openai", MODEL, "resp_previous")
    return handler, adapter


def call_handler(handler, method, parameters, *, with_functions=False, **kwargs):
    options = {
        "model": MODEL,
        "additional_parameters": parameters,
        "functions": [local_tool] if with_functions else None,
        **kwargs,
    }
    if method in {"request", "request_async"}:
        options.update(
            prompt="private new prompt",
            files=[TextDocumentFile("private attachment contents", "private.txt")],
        )
    result = getattr(handler, method)(**options)
    return asyncio.run(result) if method.endswith("_async") else result


@pytest.mark.parametrize("method", REQUEST_METHODS)
@pytest.mark.parametrize("with_functions", [False, True])
@pytest.mark.parametrize("nested", [False, True], ids=["friendly", "nested"])
@pytest.mark.parametrize("agent_count", [2, 3, "2", "3"])
def test_conflict_is_rejected_before_network_or_conversation_mutation(
    handler_and_adapter, method, with_functions, nested, agent_count
):
    handler, adapter = handler_and_adapter
    conversation = handler.the_conversation
    before = conversation.save_to_json()
    usage_before = deepcopy(conversation.usage_total)
    parameters = {"agent_count": agent_count}
    parameters.update({"reasoning": {"mode": "pro"}} if nested else {"reasoning_mode": "pro"})

    with pytest.raises(RequestValidationError) as raised:
        call_handler(handler, method, parameters, with_functions=with_functions)

    error = raised.value
    assert isinstance(error, ValueError)
    assert str(error) == ERROR_MESSAGE
    assert error.code == "incompatible_parameters"
    assert error.parameters == ("reasoning.mode", "agent_count")
    assert conversation.save_to_json() == before
    assert conversation.usage_total == usage_before
    assert conversation.continuation("openai", MODEL)["response_id"] == "resp_previous"
    assert adapter._client.mock_calls == []
    assert adapter._async_client.mock_calls == []


@pytest.mark.parametrize("method", REQUEST_METHODS)
@pytest.mark.parametrize("with_functions", [False, True])
@pytest.mark.parametrize(
    "parameters, expected_mode, expected_agents",
    [
        ({"reasoning_mode": "pro"}, "pro", 0),
        ({"reasoning_mode": "pro", "agent_count": 0}, "pro", 0),
        ({"reasoning": {"mode": "pro"}, "agent_count": "0"}, "pro", 0),
        ({"reasoning_mode": "standard", "agent_count": 2}, "standard", 2),
        ({"reasoning": {"mode": "standard"}, "agent_count": "3"}, "standard", 3),
        ({"agent_count": 2}, None, 2),
        ({"agent_count": 3}, None, 3),
        ({"agent_count": "3"}, None, 3),
    ],
)
def test_valid_alternatives_keep_requested_settings(
    handler_and_adapter, method, with_functions, parameters, expected_mode, expected_agents
):
    handler, adapter = handler_and_adapter
    parameters = deepcopy(parameters)
    parameters.update(reasoning_effort="max", verbosity="high", web_search=True)

    message = call_handler(handler, method, parameters, with_functions=with_functions)

    client = adapter._async_client if method.endswith("_async") else adapter._client
    create = client.beta.responses.create if expected_agents else client.responses.create
    create.assert_called_once()
    payload = create.call_args.kwargs
    assert payload["model"] == MODEL
    assert payload["reasoning"].get("mode") == expected_mode
    if expected_mode is None:
        assert "mode" not in payload["reasoning"]
    assert payload["reasoning"]["effort"] == "max"
    assert payload["text"]["verbosity"] == "high"
    assert {"type": "web_search"} in payload["tools"]
    if expected_agents:
        assert payload["multi_agent"] == {
            "enabled": True, "max_concurrent_subagents": expected_agents,
        }
        client.responses.create.assert_not_called()
    else:
        assert "multi_agent" not in payload
        client.beta.responses.create.assert_not_called()
    if with_functions:
        assert any(tool.get("name") == "local_tool" for tool in payload["tools"])
    assert message is handler.the_conversation.messages[-1]


@pytest.mark.parametrize("method", REQUEST_METHODS)
def test_deprecated_kwargs_are_validated(handler_and_adapter, method):
    handler, adapter = handler_and_adapter
    before = handler.the_conversation.save_to_json()
    with pytest.raises(RequestValidationError):
        call_handler(handler, method, None, reasoning_mode="pro", agent_count="3")
    assert handler.the_conversation.save_to_json() == before
    assert adapter._client.mock_calls == []
    assert adapter._async_client.mock_calls == []


@pytest.mark.parametrize("friendly_mode, rejected", [("pro", True), ("standard", False)])
def test_validation_uses_normalized_precedence(handler_and_adapter, friendly_mode, rejected):
    handler, adapter = handler_and_adapter
    parameters = {
        "reasoning_mode": friendly_mode,
        "reasoning": {"mode": "standard" if rejected else "pro"},
        "agent_count": "3",
    }
    if rejected:
        with pytest.raises(RequestValidationError):
            call_handler(handler, "request", parameters, reasoning_mode="standard")
        assert adapter._client.mock_calls == []
    else:
        call_handler(handler, "request", parameters, reasoning_mode="pro")
        payload = adapter._client.beta.responses.create.call_args.kwargs
        assert payload["reasoning"]["mode"] == "standard"


@pytest.mark.parametrize("method", REQUEST_METHODS)
@pytest.mark.parametrize("rejected", [False, True])
def test_parameters_are_normalized_once(handler_and_adapter, method, rejected):
    handler, _ = handler_and_adapter
    handler.normalizer.normalize = MagicMock(wraps=handler.normalizer.normalize)
    parameters = {"reasoning_mode": "pro", "agent_count": 3 if rejected else 0}
    if rejected:
        with pytest.raises(RequestValidationError):
            call_handler(handler, method, parameters)
    else:
        call_handler(handler, method, parameters)
    handler.normalizer.normalize.assert_called_once()


@pytest.mark.parametrize("method", ["request", "request_async"])
def test_conflicting_model_defaults_are_validated(handler_and_adapter, method):
    handler, adapter = handler_and_adapter
    handler.model_config = {MODEL: Model({
        "name": MODEL,
        "adapter": "OpenAIAdapter",
        "additional_parameters": {
            "reasoning_mode": {
                "request_key": "reasoning.mode", "default": "pro", "send_default": True,
            },
            "agent_count": {"default": "3", "send_default": True},
        },
    })}
    handler.normalizer = ParameterNormalizer(handler.model_config)
    before = handler.the_conversation.save_to_json()

    with pytest.raises(RequestValidationError):
        call_handler(handler, method, None)

    assert handler.the_conversation.save_to_json() == before
    assert adapter._client.mock_calls == []
    assert adapter._async_client.mock_calls == []


def test_rejected_normalization_cannot_mutate_shared_nested_conversation_data(handler_and_adapter):
    handler, adapter = handler_and_adapter
    conversation = handler.the_conversation
    previous_message = conversation.messages[-1]
    previous_message.provider_data["reasoning"] = {"mode": "standard", "effort": "max"}
    conversation.checkpoint("openai", MODEL, "resp_previous")
    parameters = {
        "reasoning": previous_message.provider_data["reasoning"],
        "reasoning_mode": "pro",
        "agent_count": "3",
    }
    parameters_before = deepcopy(parameters)
    conversation_before = conversation.save_to_json()

    with pytest.raises(RequestValidationError):
        call_handler(handler, "request", parameters)

    assert parameters == parameters_before
    assert conversation.save_to_json() == conversation_before
    assert conversation.continuation("openai", MODEL)["response_id"] == "resp_previous"
    assert adapter._client.mock_calls == []


@pytest.mark.parametrize(
    "method",
    ["request_llm", "request_llm_async", "request_llm_with_functions", "request_llm_with_functions_async"],
)
def test_direct_adapter_entry_points_reject_conflict(handler_and_adapter, method):
    handler, adapter = handler_and_adapter
    before = handler.the_conversation.save_to_json()
    options = {
        "model": MODEL,
        "the_conversation": handler.the_conversation,
        "additional_parameters": {"reasoning": {"mode": "pro"}, "agent_count": "3"},
        "functions": [local_tool],
    }
    with pytest.raises(RequestValidationError):
        result = getattr(adapter, method)(**options)
        if method.endswith("_async"):
            asyncio.run(result)
    assert handler.the_conversation.save_to_json() == before
    assert adapter._client.mock_calls == []
    assert adapter._async_client.mock_calls == []


def test_error_contains_only_safe_configuration_metadata(handler_and_adapter):
    handler, _ = handler_and_adapter
    with pytest.raises(RequestValidationError) as raised:
        call_handler(handler, "request", {
            "reasoning_mode": "pro", "agent_count": "3", "api_key": "private credential",
        })
    assert vars(raised.value) == {
        "code": "incompatible_parameters", "parameters": ("reasoning.mode", "agent_count"),
    }
    assert raised.value.args == (ERROR_MESSAGE,)


class UnrelatedAdapter(AdapterBase):
    """A provider whose own agent_count and reasoning mode are compatible."""

    def convert_conversation_history_to_adapter_format(self, the_conversation, **kwargs):
        return []

    def request_llm(self, *, additional_parameters, **kwargs):
        self.received_parameters = additional_parameters
        return Message(role="assistant", content="allowed")


@pytest.mark.parametrize("method", REQUEST_METHODS)
def test_rule_does_not_apply_to_other_providers(method):
    handler = APIHandler()
    model = Model({
        "name": MODEL,
        "adapter": "UnrelatedAdapter",
        "additional_parameters": {
            "reasoning_mode": {"request_key": "reasoning.mode", "send_default": False},
            "agent_count": {"send_default": False},
        },
    })
    handler.model_config = {MODEL: model}
    handler.normalizer = ParameterNormalizer(handler.model_config)
    adapter = UnrelatedAdapter()
    handler.adapters["UnrelatedAdapter"] = adapter

    message = call_handler(handler, method, {"reasoning_mode": "pro", "agent_count": "3"})

    assert message.content == "allowed"
    assert adapter.received_parameters == {"reasoning": {"mode": "pro"}, "agent_count": "3"}
