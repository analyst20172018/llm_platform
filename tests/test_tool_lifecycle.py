"""Offline tool lifecycle, response fidelity, and bounded polling regressions."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_platform.adapters.adapter_base import ResponseTimeoutError
from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.grok_adapter import GrokAdapter
from llm_platform.adapters.kimi_adapter import KimiAdapter
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.adapters.response_metadata import result_status
from llm_platform.services.conversation import Conversation, Message


def obj(data):
    if isinstance(data, dict):
        return NS(**{key: value if key in {"arguments", "input"} else obj(value)
                     for key, value in data.items()})
    if isinstance(data, list):
        return [obj(value) for value in data]
    return data


def roundtrip(conversation):
    return Conversation.read_from_json(json.loads(json.dumps(conversation.save_to_json())))


def lookup() -> str:
    """Return a local result."""
    return "local result"


def chat_response(reason="stop", calls=None):
    return NS(id="r1", usage=None, choices=[NS(finish_reason=reason, message=NS(
        content="answer", tool_calls=calls or [], reasoning_content=None,
    ))])


def chat_call():
    return NS(id="call_1", type="function", function=NS(name="lookup", arguments="{}"))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("forced", ["required", {"type": "function", "function": {"name": "lookup"}}])
def test_kimi_forces_only_first_round(asynchronous, forced):
    adapter = KimiAdapter()
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    policies = []

    def respond(**kwargs):
        policy = kwargs["tool_choice"]
        policies.append(policy)
        assert len(policies) <= 2, "forced tool choice did not relax"
        return chat_response("tool_calls", [chat_call()]) if policy != "auto" else chat_response()

    create = AsyncMock(side_effect=respond) if asynchronous else MagicMock(side_effect=respond)
    client = adapter._async_client if asynchronous else adapter._client
    client.chat.completions.create = create
    conversation = Conversation([Message("user", "question")])
    parameters = {"tool_choice": forced}
    args = ("kimi-k3", conversation, [lookup])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async(*args, additional_parameters=parameters))
    else:
        message = adapter.request_llm(*args, additional_parameters=parameters)
    assert policies == [forced, "auto"]
    assert parameters == {"tool_choice": forced}
    assert message.status == "completed"
    assert conversation.messages[1].status == "requires_action"
    assert conversation.messages[1].function_responses[0].response == {"text": "local result"}


class Events:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class AsyncEvents(Events):
    async def __aiter__(self):
        for event in self.events:
            yield event

    async def close(self):
        self.closed = True


def claude_response(reason, blocks):
    return obj({"id": "claude_1", "model": "test", "stop_reason": reason,
                "container": {"id": "container_1"}, "content": blocks,
                "usage": {"input_tokens": 2, "output_tokens": 3}})


def claude_events(reason, blocks, asynchronous):
    events = [obj({"type": "message_start", "message": {
        "id": "claude_1", "model": "test", "usage": {"input_tokens": 2},
    }})]
    for index, block in enumerate(blocks):
        # Exercise text and citation deltas, not just pre-filled blocks.
        start = dict(block)
        if start["type"] == "text":
            start["text"] = ""
            start["citations"] = []
        events.append(obj({"type": "content_block_start", "index": index, "content_block": start}))
        if block["type"] == "text":
            events.append(obj({"type": "content_block_delta", "index": index,
                               "delta": {"type": "text_delta", "text": block["text"]}}))
            for citation in block.get("citations", []):
                events.append(obj({"type": "content_block_delta", "index": index,
                                   "delta": {"type": "citations_delta", "citation": citation}}))
        events.append(obj({"type": "content_block_stop", "index": index}))
    events.append(obj({"type": "message_delta", "delta": {
        "stop_reason": reason, "container": {"id": "container_1"}}, "usage": {"output_tokens": 3}}))
    return (AsyncEvents if asynchronous else Events)(events)


def make_claude(responses, asynchronous):
    adapter = AnthropicAdapter()
    adapter.model_config = {"test": {"adaptive_thinking": False}}
    adapter.correct_max_tokens = MagicMock(side_effect=lambda model, history, maximum, tools: maximum)
    adapter.correct_max_tokens_async = AsyncMock(side_effect=lambda model, history, maximum, tools: maximum)
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    (adapter._async_client if asynchronous else adapter._client).beta.messages.create = create
    return adapter, create


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_local", [False, True])
def test_claude_pause_resumes_all_tools_blocks_container_and_citations(asynchronous, stream, with_local):
    citation = {"type": "web_search_result_location", "url": "https://example.com", "title": "Source",
                "cited_text": "evidence", "encrypted_index": "opaque"}
    paused = [{"type": "redacted_thinking", "data": "opaque"},
              {"type": "text", "text": "partial", "citations": [citation]},
              {"type": "server_tool_use", "id": "server_1", "name": "bash_code_execution", "input": {}}]
    final = [{"type": "bash_code_execution_tool_result", "tool_use_id": "server_1",
              "content": {"stdout": "42", "file_id": "file_1"}},
             {"type": "text", "text": "done", "citations": [citation]}]
    factory = (lambda reason, blocks: claude_events(reason, blocks, asynchronous)) if stream else claude_response
    responses = [factory("pause_turn", paused), factory("end_turn", final)]
    adapter, create = make_claude(responses, asynchronous)
    conversation = Conversation([Message("user", "question")])
    options = {"web_search": True, "code_execution": True, "max_tokens": 22000 if stream else 100}
    args = ("test", conversation, [lookup] if with_local else [])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async(*args, additional_parameters=options))
    else:
        message = adapter.request_llm(*args, additional_parameters=options)
    assert create.call_count == 2
    expected_names = {"web_search", "code_execution"} | ({"lookup"} if with_local else set())
    assert all({tool["name"] for tool in call.kwargs["tools"]} == expected_names for call in create.call_args_list)
    followup = create.call_args.kwargs
    assert followup["container"] == "container_1"
    assert followup["messages"][-1] == {"role": "assistant", "content": paused}
    assert message.content == "done"
    assert message.status == "completed"
    assert message.citations[0]["source"] == citation
    assert message.hosted_tool_results == final[:1]
    restored = roundtrip(conversation)
    assert restored.messages[1].status == "paused"
    assert restored.messages[1].citations[0]["source"] == citation
    assert restored.messages[-1].hosted_tool_results == final[:1]
    assert conversation.usage_total["completion_tokens"] == 6
    if stream:
        assert all(response.closed for response in responses)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_claude_pause_round_and_time_limits_preserve_partial_history(monkeypatch, asynchronous):
    import llm_platform.adapters.anthropic_adapter as module
    monkeypatch.setattr(module, "MAX_TOOL_ROUNDS", 2)
    response = claude_response("pause_turn", [{"type": "text", "text": "partial"}])
    adapter, create = make_claude([response] * 3, asynchronous)
    conversation = Conversation([Message("user", "question")])

    def request():
        if asynchronous:
            return asyncio.run(adapter.request_llm_async("test", conversation))
        return adapter.request_llm("test", conversation)

    with pytest.raises(RuntimeError, match="maximum tool-calling rounds"):
        request()
    assert create.call_count == 2
    assert [m.status for m in conversation.messages[1:]] == ["paused", "paused"]
    now = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def delayed(**kwargs):
        now[0] = 2
        return response

    create.side_effect = delayed
    create.reset_mock()
    adapter.RESPONSE_TIMEOUT_SECONDS = 1
    with pytest.raises(ResponseTimeoutError) as error:
        request()
    assert create.call_count == 1
    assert error.value.response.id == "claude_1"
    assert conversation.messages[-1].content == "partial"


@pytest.mark.parametrize("reason,status", [("end_turn", "completed"), ("max_tokens", "incomplete"),
                                           ("refusal", "refused"), ("new_reason", "unknown")])
def test_claude_finish_status(reason, status):
    adapter, _ = make_claude([claude_response(reason, [{"type": "text", "text": "partial"}])], False)
    message = adapter.request_llm("test", Conversation())
    assert (message.status, message.finish_reason) == (status, reason)


def openai_response(status, output=None):
    return obj({"id": "resp_1", "model": "test", "status": status, "usage": None,
                "output": output or [], "error": {"code": "server_error"} if status == "failed" else None,
                "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", ["completed", "failed", "incomplete", "cancelled"])
def test_openai_status_citations_hosted_results_and_failed_tool_guard(asynchronous, status):
    citation = {"type": "url_citation", "url": "https://example.com", "title": "Source",
                "start_index": 0, "end_index": 7}
    output = [{"type": "message", "content": [{"type": "output_text", "text": "partial", "annotations": [citation]}]},
              {"type": "web_search_call", "id": "search_1", "status": "completed", "action": {"sources": [{"url": "https://example.com"}]}}]
    if status != "completed":
        output.append({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": "{"})
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    response = openai_response(status, output)
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    (adapter._async_client if asynchronous else adapter._client).responses.create = create
    conversation = Conversation([Message("user", "question")])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async("test", conversation, [lookup]))
    else:
        message = adapter.request_llm("test", conversation, [lookup])
    assert create.call_count == 1
    assert message.content == "partial"
    assert message.status == status
    assert message.citations[0]["source"] == citation
    assert message.citations[0]["start_index"] == 0
    assert message.hosted_tool_results == output[1:2]
    assert roundtrip(conversation).messages[-1].error == message.error
    if status == "incomplete":
        assert message.incomplete_details == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("agent", [None, "antigravity", "deep_research"])
def test_google_failed_results_preserved_without_tool_execution(asynchronous, agent):
    adapter = GoogleAdapter()
    adapter.model_config = {"test": {"agent_type": agent, "background_mode": bool(agent), "uses_thinking_level": False}}
    adapter._client = MagicMock()
    steps = [{"type": "model_output", "content": [{"type": "text", "text": "partial", "annotations": [
        {"type": "url_citation", "url": "https://example.com", "start_index": 0, "end_index": 7}]}]},
        {"type": "code_execution_result", "result": {"stdout": "42", "files": [{"uri": "file_1"}]}},
        {"type": "function_call", "id": "call_1", "name": "lookup", "arguments": "{"}]
    response = obj({"id": "i1", "status": "failed", "steps": steps, "error": {"code": "failure"}})
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    client = adapter._client.aio if asynchronous else adapter._client
    client.interactions.create = create
    conversation = Conversation([Message("user", "question")])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async("test", conversation, [lookup]))
    else:
        message = adapter.request_llm("test", conversation, [lookup])
    assert create.call_count == 1
    assert message.status == "failed"
    assert message.content == "partial"
    assert message.error == {"code": "failure"}
    assert message.hosted_tool_results == steps[1:2]
    assert roundtrip(conversation).messages[-1].citations == message.citations


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["openai", "google", "research"])
def test_polling_deadline_retains_last_response(monkeypatch, asynchronous, provider):
    import llm_platform.adapters.adapter_base as base
    now = [0.0]
    monkeypatch.setattr(base.time, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += seconds

    async def async_sleep(seconds):
        sleep(seconds)

    monkeypatch.setattr(base.time, "sleep", sleep)
    monkeypatch.setattr(asyncio, "sleep", async_sleep)
    adapter = OpenAIAdapter() if provider == "openai" else GoogleAdapter()
    adapter.RESPONSE_TIMEOUT_SECONDS = 3
    response = NS(id="pending_1", status="in_progress")
    method = {"openai": "_poll_background_response", "google": "_poll_agent_interaction",
              "research": "_poll_deep_research_interaction"}[provider]
    with pytest.raises(ResponseTimeoutError) as error:
        if asynchronous:
            asyncio.run(getattr(adapter, method + "_async")(response))
        else:
            getattr(adapter, method)(response)
    assert now[0] == 3
    assert error.value.response is response
    assert error.value.response_id == "pending_1"


@pytest.mark.parametrize("reason,status", [("stop", "completed"), ("length", "incomplete"),
                                           ("content_filter", "refused"), ("other", "unknown")])
def test_chat_finish_metadata_roundtrips(reason, status):
    message = OpenRouterAdapter()._message_from_response("test", chat_response(reason))
    restored = roundtrip(Conversation([message])).messages[0]
    assert (restored.status, restored.finish_reason) == (status, reason)
    assert restored.replay_data("openrouter", "test")


def test_grok_keeps_hosted_calls_remote_and_exposes_citations():
    adapter = GrokAdapter()
    adapter._client = MagicMock()
    hosted = NS(id="search_1", type=2, status=1, function=NS(name="web_search", arguments="{}"))
    response = NS(id="g1", content="answer", finish_reason="REASON_STOP", tool_calls=[hosted],
                  usage=None, citations=["https://example.com"],
                  inline_citations=[NS(start_index=0, end_index=6, web_citation=NS(url="https://example.com"))])
    adapter._client.chat.create.return_value.sample.return_value = response
    conversation = Conversation([Message("user", "question")])
    message = adapter.request_llm("test", conversation, [lookup])
    assert adapter._client.chat.create.call_count == 1
    assert message.status == "completed"
    assert message.hosted_tool_results[0]["id"] == "search_1"
    assert message.citations[1]["start_index"] == 0
    assert message.citations[1]["url"] == "https://example.com"


def test_unknown_status_is_not_assumed_completed():
    assert result_status("new_provider_state") == "unknown"
    assert result_status("new_provider_state", has_calls=True) == "unknown"
    assert Message("assistant", "old answer").status == "unknown"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_claude_local_tool_then_hosted_pause_keeps_every_result(asynchronous, stream):
    factory = (lambda reason, blocks: claude_events(reason, blocks, asynchronous)) if stream else claude_response
    first = [{"type": "server_tool_use", "id": "s1", "name": "code_execution", "input": {}},
             {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}}]
    paused = [{"type": "code_execution_tool_result", "tool_use_id": "s1", "content": {"stdout": "42"}}]
    responses = [factory("tool_use", first), factory("pause_turn", paused),
                 factory("end_turn", [{"type": "text", "text": "done"}])]
    adapter, create = make_claude(responses, asynchronous)
    conversation = Conversation([Message("user", "question")])
    options = {"code_execution": True, "max_tokens": 22000 if stream else 100}
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async("test", conversation, [lookup], additional_parameters=options))
    else:
        message = adapter.request_llm("test", conversation, [lookup], additional_parameters=options)
    assert message.status == "completed"
    assert [m.status for m in conversation.messages[1:]] == ["requires_action", "paused", "completed"]
    second_history = create.call_args_list[1].kwargs["messages"]
    assert second_history[-2]["content"] == first
    assert second_history[-1]["content"][0]["tool_use_id"] == "c1"
    assert conversation.messages[1].function_responses[0].response == {"text": "local result"}
    assert roundtrip(conversation).messages[2].hosted_tool_results == paused


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["length", "content_filter", "unrecognized"])
def test_kimi_does_not_execute_truncated_or_refused_calls(asynchronous, reason):
    adapter = KimiAdapter()
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    response = chat_response(reason, [chat_call()])
    response.choices[0].message.tool_calls[0].function.arguments = "{"
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    (adapter._async_client if asynchronous else adapter._client).chat.completions.create = create
    conversation = Conversation([Message("user", "question")])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async("kimi-k3", conversation, [lookup]))
    else:
        message = adapter.request_llm("kimi-k3", conversation, [lookup])
    assert create.call_count == 1
    assert not message.can_execute_tools
    assert message.function_calls[0].arguments == "{"


def test_openai_refusal_is_visible():
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._client.responses.create.return_value = openai_response("completed", [
        {"type": "message", "content": [{"type": "refusal", "refusal": "Cannot answer"}]},
    ])
    message = adapter.request_llm("test", Conversation())
    assert (message.status, message.content) == ("refused", "Cannot answer")


def test_openai_subagent_refusal_does_not_override_completed_root_answer():
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._client.responses.create.return_value = openai_response("completed", [
        {"type": "message", "agent": {"agent_name": "/root/helper"},
         "content": [{"type": "refusal", "refusal": "Cannot answer"}]},
        {"type": "message", "agent": {"agent_name": "/root"}, "phase": "final_answer",
         "content": [{"type": "output_text", "text": "answer"}]},
    ])
    message = adapter.request_llm("test", Conversation())
    assert (message.status, message.content) == ("completed", "answer")


def test_mistral_reference_chunk_is_exposed_and_replayed():
    from mistralai.client.models import AssistantMessage, ReferenceChunk, TextChunk
    from llm_platform.adapters.mistral_adapter import MistralAdapter

    assistant = AssistantMessage(content=[TextChunk(type="text", text="answer"),
                                          ReferenceChunk(type="reference", reference_ids=["source_1"])])
    response = NS(id="m1", choices=[NS(message=assistant, finish_reason="stop")], usage=None)
    message = MistralAdapter()._message_from_response("test", response)
    assert message.content == "answer"
    assert message.citations[0]["source"]["reference_ids"] == ["source_1"]
    restored = roundtrip(Conversation([message])).messages[0]
    assert restored.replay_data("mistral", "test")["message"]["content"][1]["type"] == "reference"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("agent", [None, "antigravity"])
def test_google_round_limit_does_not_send_an_unobserved_extra_request(monkeypatch, asynchronous, agent):
    import llm_platform.adapters.google_adapter as module
    monkeypatch.setattr(module, "MAX_TOOL_ROUNDS", 1)
    adapter = GoogleAdapter()
    adapter.model_config = {"test": {"agent_type": agent, "background_mode": False, "uses_thinking_level": False}}
    adapter._client = MagicMock()
    response = obj({"id": "i1", "status": "requires_action", "steps": [
        {"type": "function_call", "id": "c1", "name": "lookup", "arguments": {}},
    ]})
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    (adapter._client.aio if asynchronous else adapter._client).interactions.create = create
    conversation = Conversation([Message("user", "question")])
    with pytest.raises(RuntimeError, match="maximum tool-calling rounds"):
        if asynchronous:
            asyncio.run(adapter.request_llm_async("test", conversation, [lookup]))
        else:
            adapter.request_llm("test", conversation, [lookup])
    assert create.call_count == 1
    assert conversation.messages[-1].function_responses[0].response == {"text": "local result"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_local", [False, True])
def test_openai_container_result_downloaded_once_and_citation_persisted(asynchronous, with_local):
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    citation = {"type": "container_file_citation", "container_id": "container_1",
                "file_id": "file_1", "filename": "report.txt", "start_index": 0, "end_index": 6}
    response = openai_response("completed", [{"type": "message", "content": [
        {"type": "output_text", "text": "report", "annotations": [citation]},
    ]}])
    client = adapter._async_client if asynchronous else adapter._client
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    retrieve = AsyncMock(return_value=NS(content=b"file result")) if asynchronous else MagicMock(return_value=NS(content=b"file result"))
    client.responses.create = create
    client.containers.files.content.retrieve = retrieve
    conversation = Conversation([Message("user", "question")])
    if asynchronous:
        message = asyncio.run(adapter.request_llm_async("test", conversation, [lookup] if with_local else []))
    else:
        message = adapter.request_llm("test", conversation, [lookup] if with_local else [])
    assert retrieve.call_count == 1
    assert message.files[0].text == "file result"
    assert roundtrip(conversation).messages[-1].citations[0]["source"] == citation
