"""Real SDK serialization through httpx transports; no provider requests."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import anthropic
import httpx
import httpx2
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import pytest

from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.wiro_gateway_adapter import WiroGatewayAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message


GATEWAY = "qwen/qwen3-8-27b-uncensored"


def openai_response():
    return {"id": "resp_test", "object": "response", "created_at": 1,
            "model": "gpt-5.6-luna", "status": "completed", "output": [
                {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": '{"answer": "ok"}', "annotations": []}]}],
            "usage": {"input_tokens": 2000, "output_tokens": 10, "total_tokens": 2010,
                      "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 500},
                      "output_tokens_details": {"reasoning_tokens": 3}}}


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("parse", [False, True])
def test_openai_cache_sdk_serialization_and_replay(asynchronous, parse):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response())

    handler = APIHandler(system_prompt="Reusable instructions")
    adapter = handler.get_adapter("gpt-5.6-luna")
    adapter._client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    adapter._async_client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    conversation = handler.the_conversation
    user = Message("user", "Reusable context", id="stable-context")
    conversation.messages.append(user)
    conversation.checkpoint("openai", "gpt-5.6-luna", "old_response")
    parameters = {"max_tokens": 64, "prompt_cache_key": "cache-test",
                  "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
                  "prompt_cache_breakpoints": ["system", user.id]}
    if parse:
        parameters["structured_output"] = Answer
    original = deepcopy(parameters)
    for _ in range(2):
        if asynchronous:
            message = asyncio.run(handler.request_async("gpt-5.6-luna", "Answer", additional_parameters=parameters))
        else:
            message = handler.request("gpt-5.6-luna", "Answer", additional_parameters=parameters)
    for payload in requests:
        assert "previous_response_id" not in payload
        assert "instructions" not in payload
        assert "prompt_cache_breakpoints" not in payload
        assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        assert payload["prompt_cache_key"] == "cache-test"
        assert payload["input"][0]["role"] == "developer"
        assert payload["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert payload["input"][1]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert parameters == original
    assert user.content == "Reusable context"
    assert message.usage["prompt_tokens"] == 2000  # cache counts are subsets, not added again
    assert message.usage["cache_creation_tokens"] == 500
    restored = Conversation.read_from_json(json.loads(json.dumps(conversation.save_to_json())))
    assert restored.usage_total["cache_read_tokens"] == 2000
    assert restored.usage_total["cache_creation_tokens"] == 1000
    assert restored.usage_last["provider_usage"]["output_tokens_details"]["reasoning_tokens"] == 3
    # Removing the system boundary must not reuse remote developer instructions
    # alongside the ordinary top-level instructions and duplicate them.
    restored.messages.append(Message("user", "No cache boundary this time"))
    next_request = adapter._create_parameters_for_calling_llm("gpt-5.6-luna", restored, {})
    assert "previous_response_id" not in next_request
    assert next_request["instructions"] == "Reusable instructions"
    assert not any(item.get("role") == "developer" for item in next_request["input"])
    adapter.client.close()
    asyncio.run(adapter.async_client.close())


@pytest.mark.parametrize("parameters", [
    {"prompt_cache_options": {"mode": "bad"}},
    {"prompt_cache_options": {"ttl": "24h"}},
    {"prompt_cache_options": {"typo": True}},
    {"prompt_cache_breakpoints": "system"},
    {"prompt_cache_breakpoints": ["missing"]},
    {"prompt_cache_breakpoints": ["system", "system"]},
    {"prompt_cache_breakpoints": ["a", "b", "c", "d"]},
    {"prompt_cache_breakpoints": ["assistant"]},
])
def test_invalid_cache_policy_fails_before_sdk(parameters):
    conversation = Conversation([Message("assistant", "answer", id="assistant")])
    adapter = OpenAIAdapter()
    with pytest.raises(ValueError):
        adapter._create_parameters_for_calling_llm("gpt-5.6-luna", conversation, parameters)
    assert adapter._client is None


def test_cache_policy_without_breakpoints_keeps_continuation():
    conversation = Conversation([Message("user", "first")])
    conversation.checkpoint("openai", "gpt-5.6-luna", "previous")
    conversation.messages.append(Message("user", "next"))
    result = OpenAIAdapter()._create_parameters_for_calling_llm(
        "gpt-5.6-luna", conversation, {"prompt_cache_options": {"mode": "explicit"}})
    assert result["previous_response_id"] == "previous"
    assert len(result["input"]) == 1


def test_missing_usage_remains_unknown():
    usage = OpenAIAdapter()._parse_response(NS(model="test", output=[], usage=None))[-1]
    assert usage["cache_read_tokens"] is None
    assert usage["cache_creation_tokens"] is None


def local_lookup() -> str:
    """Return a local value."""
    return "local"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_claude_new_hosted_tools_sdk_serialization(asynchronous):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if request.url.path.endswith("count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 2})
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 2, "output_tokens": 3,
                      "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50},
        })

    handler = APIHandler()
    adapter = handler.get_adapter("claude-sonnet-5")
    adapter._client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(respond)))
    adapter._async_client = anthropic.AsyncAnthropic(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)))
    parameters = {"max_tokens": 1024, "web_search": True, "code_execution": True,
                  "web_search_options": {"max_uses": 1, "response_inclusion": "excluded",
                                         "allowed_domains": ["python.org"]}}
    original = deepcopy(parameters)
    if asynchronous:
        message = asyncio.run(handler.request_async("claude-sonnet-5", "test", [local_lookup], additional_parameters=parameters))
    else:
        message = handler.request("claude-sonnet-5", "test", [local_lookup], additional_parameters=parameters)
    tools = {tool["name"]: tool for tool in payloads[-1]["tools"]}
    assert tools["web_search"] == {"name": "web_search", "type": "web_search_20260318",
                                   **parameters["web_search_options"]}
    assert tools["code_execution"]["type"] == "code_execution_20260521"
    assert "local_lookup" in tools
    assert "web_search_options" not in payloads[-1]
    assert parameters == original
    assert message.usage["prompt_tokens"] == 152
    assert message.usage["provider_usage"]["cache_creation_input_tokens"] == 50
    adapter.client.close()
    asyncio.run(adapter.async_client.close())


@pytest.mark.parametrize("options", [
    {"type": "override"}, {"allowed_domains": [], "blocked_domains": []},
    {"max_uses": 0}, {"max_uses": True}, {"response_inclusion": "none"},
])
def test_claude_search_option_validation(options):
    with pytest.raises(ValueError):
        AnthropicAdapter()._build_tools(None, {"web_search": True, "web_search_options": options})


def gateway_contract():
    return {"id": GATEWAY, "object": "model", "created": 0, "owned_by": "qwen",
            "capabilities": {"endpoints": ["chat"], "input_modalities": ["text"], "function_tools": True,
                             "structured_output_modes": ["json_object", "json_schema"],
                             "strict_json_schema": True},
            "top_provider": {"max_completion_tokens": 4096}}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("secret", [None, "secret"])
def test_wiro_gateway_discovery_auth_and_local_tool_roundtrip(monkeypatch, asynchronous, secret):
    monkeypatch.setenv("WIRO_API_KEY", "test-key")
    if secret:
        monkeypatch.setenv("WIRO_API_SECRET", secret)
    else:
        monkeypatch.delenv("WIRO_API_SECRET", raising=False)
    requests, payloads = [], []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=gateway_contract())
        payloads.append(json.loads(request.content))
        first = len(payloads) == 1
        message = {"role": "assistant", "content": None if first else "done"}
        if first:
            message["tool_calls"] = [{"id": "call_1", "type": "function",
                                      "function": {"name": "local_lookup", "arguments": "{}"}}]
        return httpx.Response(200, json={
            "id": "chat_test", "created": 1, "object": "chat.completion", "model": GATEWAY,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if first else "stop"}],
        })

    handler = APIHandler()
    adapter = handler.get_adapter(GATEWAY)
    assert adapter._client is None
    # Use actual client constructors to exercise the gateway credential implementation.
    adapter._client = adapter.client.with_options(http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    adapter._async_client = adapter.async_client.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    params = {"max_tokens": 512, "tool_choice": "required"}
    if asynchronous:
        message = asyncio.run(handler.request_async(GATEWAY, "test", [local_lookup], additional_parameters=params))
    else:
        message = handler.request(GATEWAY, "test", [local_lookup], additional_parameters=params)
    assert message.status == "completed"
    assert [r.method for r in requests] == ["GET", "POST", "POST"]
    assert requests[0].url.path == f"/v1/models/{GATEWAY}"
    assert all(r.url.host == "llm.wiro.ai" for r in requests)
    for request in requests:
        assert request.headers["authorization"] == "Bearer test-key" + (":secret" if secret else "")
        assert "x-nonce" not in request.headers and "x-signature" not in request.headers
    assert payloads[0]["max_completion_tokens"] == 512
    assert "max_tokens" not in payloads[0]
    assert payloads[1]["tool_choice"] == "auto"
    assert payloads[1]["messages"][-1]["tool_call_id"] == "call_1"
    adapter.client.close()
    asyncio.run(adapter.async_client.close())


@pytest.mark.parametrize("change,functions,parameters", [
    ({"endpoints": ["responses"]}, None, {}),
    ({"function_tools": False}, [local_lookup], {}),
    ({"structured_output_modes": []}, None, {"structured_output": True}),
    ({"strict_json_schema": False}, None, {"structured_output": Answer}),
    ({}, None, {"max_completion_tokens": 10000}),
])
def test_wiro_rejects_ineligible_contracts(change, functions, parameters):
    contract = gateway_contract()
    contract["capabilities"].update(change)
    with pytest.raises(ValueError):
        WiroGatewayAdapter._validate_contract(GATEWAY, contract, functions, parameters)


def test_new_parameters_are_scoped_to_provider():
    handler = APIHandler()
    result = handler._prepare_additional_parameters("kimi-k3", {
        "web_search_options": {"max_uses": 1}, "prompt_cache_options": {"mode": "explicit"}})
    assert "web_search_options" not in result and "prompt_cache_options" not in result


def test_smoke_requires_explicit_scenario_and_credentials(monkeypatch, tmp_path):
    from llm_platform.scripts.provider_smoke import main
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = tmp_path / "smoke.json"
    assert main(["--run-live", "openai-cache", "--report", str(report)]) == 1
    assert json.loads(report.read_text())["passed"] is False


@pytest.mark.parametrize("cache_read,exit_code", [(0, 1), (1500, 0)])
def test_smoke_cache_requires_observed_read_and_retains_report(monkeypatch, tmp_path, cache_read, exit_code):
    from llm_platform.scripts import provider_smoke
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        response = openai_response()
        response["usage"]["input_tokens_details"]["cached_tokens"] = cache_read
        return httpx.Response(200, json=response)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    handler = APIHandler()
    adapter = handler.get_adapter("gpt-5.6-luna")
    adapter._client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(provider_smoke, "APIHandler", lambda: handler)
    report = tmp_path / "smoke.json"
    assert provider_smoke.main(["--run-live", "openai-cache", "--report", str(report)]) == exit_code
    result = json.loads(report.read_text())
    assert len(result["responses"]) == len(calls) == 2
    assert result["passed"] == (exit_code == 0)
    assert calls[0]["max_output_tokens"] == 64
    assert "Reusable instructions" not in report.read_text()


@pytest.mark.parametrize("blocks", [
    [{"content": {"error_code": "unavailable"}}],
    [{"content": [{"return_code": 1}]}],
])
def test_smoke_detects_embedded_hosted_failures(blocks):
    from llm_platform.scripts.provider_smoke import hosted_failure
    assert hosted_failure(blocks)
