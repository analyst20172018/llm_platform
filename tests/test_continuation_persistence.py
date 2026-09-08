"""Offline regressions for provider state and the payload sent after JSON restore."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_platform.adapters.anthropic_adapter import AnthropicAdapter, ClaudeStreamProcessor
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.adapters.mistral_adapter import MistralAdapter
from llm_platform.adapters.serializers import provider_dump
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import (
    Conversation, Message, FunctionCall, FunctionResponse, ThinkingResponse,
)
from llm_platform.services.files import (
    ImageFile, VideoFile, WordDocumentFile, PowerPointDocumentFile,
    ExcelDocumentFile, PDFDocumentFile, AudioFile, MediaFile, TextDocumentFile,
)


def restore(conversation):
    return Conversation.read_from_json(json.loads(json.dumps(conversation.save_to_json())))


def obj(value):
    if isinstance(value, dict):
        return NS(**{k: v if k in ("input", "arguments") else obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [obj(v) for v in value]
    return value


def openai_response(response_id="resp_1", output=None):
    return NS(id=response_id, model="test", output=output or [], usage=None)


def google_response(response_id="interaction_1", steps=None, environment_id=None):
    return NS(id=response_id, steps=steps or [], environment_id=environment_id,
              usage=None, status="completed", output_text="done")


def make_openai():
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    return adapter


def make_google(agent_type=None):
    adapter = GoogleAdapter()
    adapter.model_config = {"test": {"background_mode": False, "agent_type": agent_type,
                                     "uses_thinking_level": False}}
    adapter._client = MagicMock()
    return adapter


def test_lossless_versioned_round_trip_and_non_mutating_tool_results():
    source = {"value": 42, "files": [{"type": "image", "source": {
        "type": "base64", "format": "png", "data": "aW1hZ2U="}}]}
    result = FunctionResponse("weather", source, id="item_result", call_id="call_123")
    assert "files" in source
    audio = AudioFile.__new__(AudioFile)
    MediaFile.__init__(audio, b"already converted mp3", "original.wav")
    files = [cls.from_bytes(b"bytes", name) for cls, name in [
        (ImageFile, "image.png"), (VideoFile, "video.mp4"),
        (WordDocumentFile, "file.docx"), (PowerPointDocumentFile, "file.pptx"),
        (ExcelDocumentFile, "file.xlsx"), (PDFDocumentFile, "file.pdf"),
    ]] + [audio, TextDocumentFile("text", "file.txt"), ImageFile(b"", "empty.png")]
    message = Message(
        role="assistant", content="answer", id="resp_123", provider="openai", model="test",
        provider_data={"output": [{"type": "reasoning", "encrypted_content": "opaque"}]},
        thinking_responses=[ThinkingResponse("summary", "rs_1")],
        function_calls=[FunctionCall("fc_123", "weather", '{ "x": 1 }', "call_123"),
                        FunctionCall("fc_456", "weather", "{}", "call_456")],
        function_responses=[result, FunctionResponse("weather", {}, call_id="call_456")],
        files=files, additional_responses=[{"citation": {"url": "https://example.com"}}],
        usage={"prompt_tokens": 1, "cache_read_tokens": 2},
    )
    conversation = Conversation([message], "system")
    conversation.checkpoint("openai", "test", message.id)
    saved = conversation.save_to_json()
    restored = restore(conversation)
    assert restored.save_to_json() == saved
    assert restored.continuation("openai", "test")["response_id"] == "resp_123"
    assert [type(f) for f in restored.messages[0].files] == [type(f) for f in files]
    assert saved["version"] == 2
    saved["messages"][0]["function_responses"][0]["response"]["value"] = 0
    assert result.response["value"] == 42


def test_legacy_persistence_never_infers_provider_from_an_id():
    data = {"messages": [{"role": "assistant", "content": "old", "id": "resp_old",
                          "function_calls": [{"id": "call_old", "name": "f", "arguments": {}}]}]}
    conversation = Conversation.read_from_json(data)
    assert conversation.messages[0].function_calls[0].call_id == "call_old"
    assert conversation.continuation("openai", "test") is None
    assert "previous_response_id" not in make_openai()._create_parameters_for_calling_llm("test", conversation)
    with pytest.raises(ValueError, match="version"):
        Conversation.read_from_json({"version": 999})


@pytest.mark.parametrize("change", ["edit", "insert", "delete", "system", "file", "clear"])
def test_checkpoint_rejects_changed_prefix(change):
    conversation = Conversation([Message("user", "question", files=[ImageFile(b"image", "a.png")]),
                                 Message("assistant", "answer")], "system")
    conversation.checkpoint("openai", "test", "resp_1")
    conversation.messages.append(Message("user", "unsent"))
    if change == "edit":
        conversation.messages[0].content = "changed"
    elif change == "insert":
        conversation.messages.insert(0, Message("user", "inserted"))
    elif change == "delete":
        del conversation.messages[0]
    elif change == "system":
        conversation.system_prompt = "changed"
    elif change == "file":
        conversation.messages[0].files[0].data = b"changed"
    else:
        conversation.clear()
    assert conversation.continuation("openai", "test") is None


def test_openai_delta_includes_intervening_foreign_turns_and_all_unsent_messages():
    adapter = make_openai()
    adapter._client.responses.create.return_value = openai_response()
    conversation = Conversation([Message("user", "first")])
    adapter.request_llm("test", conversation)
    conversation.messages.extend([Message("assistant", "foreign", id="google_id", provider="google"),
                                  Message("user", "unsent 1"), Message("user", "unsent 2")])
    params = adapter._create_parameters_for_calling_llm("test", restore(conversation))
    assert params["previous_response_id"] == "resp_1"
    assert [item["content"][0]["text"] for item in params["input"]] == ["foreign", "unsent 1", "unsent 2"]
    foreign = Conversation([Message("assistant", "foreign", id="google_id"), Message("user", "hello")])
    assert "previous_response_id" not in adapter._create_parameters_for_calling_llm("test", foreign)


class MissingState(Exception):
    status_code = 404


@pytest.mark.parametrize("asynchronous", [False, True])
def test_openai_expired_state_replays_exact_items_and_results(asynchronous):
    native = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
              {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": '{ "x": 1 }'}]
    conversation = Conversation([Message("assistant", "", id="resp_1", provider="openai", model="test",
                                         provider_data={"output": native})])
    conversation.checkpoint("openai", "test", "resp_1")
    conversation.messages.append(Message("user", "", function_responses=[FunctionResponse("f", {"ok": True}, call_id="call_1")]))
    conversation = restore(conversation)
    adapter = make_openai()
    responses = [MissingState("previous_response_id not found"), openai_response("resp_2")]
    if asynchronous:
        create = AsyncMock(side_effect=responses)
        adapter._async_client.responses.create = create
        asyncio.run(adapter.request_llm_async("test", conversation))
    else:
        create = MagicMock(side_effect=responses)
        adapter._client.responses.create = create
        adapter.request_llm("test", conversation)
    assert create.call_count == 2
    replay = create.call_args.kwargs
    assert "previous_response_id" not in replay
    assert replay["input"][:2] == native
    assert replay["input"][2]["call_id"] == "call_1"
    assert conversation.continuation("openai", "test")["response_id"] == "resp_2"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("agent_type", [None, "antigravity"])
def test_google_continuation_round_trip_and_environment(asynchronous, agent_type):
    adapter = make_google(agent_type)
    responses = [google_response("i1", environment_id="env_1"), google_response("i2")]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    if asynchronous:
        adapter._client.aio.interactions.create = create
    else:
        adapter._client.interactions.create = create
    conversation = Conversation([Message("user", "first")])
    def request():
        if asynchronous:
            return asyncio.run(adapter.request_llm_async("test", conversation))
        return adapter.request_llm("test", conversation)
    request()
    conversation = restore(conversation)
    conversation.messages.extend([Message("user", "second"), Message("user", "third")])
    request()
    params = create.call_args.kwargs
    assert params["previous_interaction_id"] == "i1"
    assert [item["content"][0]["text"] for item in params["input"]] == ["second", "third"]
    if agent_type:
        assert create.call_args_list[0].kwargs["environment"] == "remote"
        assert params["environment"] == "env_1"
        assert restore(conversation).continuation("google", "test")["environment_id"] == "env_1"


def test_google_exact_thought_and_hosted_tool_replay_is_provider_scoped():
    steps = [{"type": "thought", "signature": "sig", "summary": []},
             {"type": "google_search_call", "id": "search_1", "signature": "search_sig", "arguments": {"q": "x"}},
             {"type": "function_call", "id": "call_1", "name": "f", "arguments": {"x": 1}}]
    adapter = make_google()
    message = adapter._parse_interaction_response(google_response(steps=obj(steps)), "test")
    conversation = restore(Conversation([message, Message("function", "", function_responses=[FunctionResponse("f", {}, call_id="call_1")])]))
    replay = adapter._build_input_from_conversation(conversation, "another-gemini")
    assert replay[:3] == steps
    assert replay[-1]["call_id"] == "call_1"
    openai = make_openai().convert_conversation_history_to_adapter_format(conversation.messages, "test")
    assert "sig" not in json.dumps(openai)


def test_antigravity_explicit_reset_and_agent_config_survive_facade_normalization():
    model = "antigravity-preview-05-2026"
    options = {"new_environment": True, "agent_config": {"max_total_tokens": 50000}}
    normalized = APIHandler()._prepare_additional_parameters(model, options)
    adapter = GoogleAdapter()
    conversation = Conversation([Message("assistant", "done")])
    conversation.checkpoint("google", model, "i1", environment_id="env_1")
    kwargs = adapter._build_antigravity_kwargs(model, conversation, [], normalized)
    assert kwargs["agent_config"] == {"type": "antigravity", "max_total_tokens": 50000}
    assert adapter._continuation_input(conversation, model, antigravity=True)["environment"] == "remote"


def test_openrouter_opaque_reasoning_survives_save_and_replay():
    native = {"role": "assistant", "content": "answer", "reasoning": "visible",
              "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque", "index": 0}]}
    adapter = OpenRouterAdapter()
    response = NS(id="r1", choices=[NS(message=obj(native))], usage=None)
    message = adapter._message_from_response("test", response)
    conversation = restore(Conversation([message]))
    history, _ = adapter.convert_conversation_history_to_adapter_format(conversation, "test")
    assert history[1] == native
    assert message.thinking_responses[0].content == "visible"
    history, _ = adapter.convert_conversation_history_to_adapter_format(conversation, "other-model")
    assert "reasoning_details" not in history[1]


def test_claude_preserves_all_blocks_and_container_without_synthesizing_signatures():
    blocks = [{"type": "redacted_thinking", "data": "opaque"},
              {"type": "thinking", "thinking": "reason", "signature": "sig"},
              {"type": "text", "text": "answer", "citations": [{"type": "web_search_result_location", "url": "https://example.com"}]},
              {"type": "server_tool_use", "id": "server_1", "name": "web_search", "input": {"q": "x"}},
              {"type": "tool_use", "id": "tool_1", "name": "f", "input": {}}]
    adapter = AnthropicAdapter()
    response = NS(id="msg_1", model="test", usage=NS(input_tokens=1, output_tokens=2),
                  stop_reason="tool_use", content=obj(blocks), container=NS(id="container_1"))
    processor = adapter._parse_non_streaming_response(response)
    conversation = Conversation()
    adapter._append_tool_exchange_message(processor, conversation, [({"id": "tool_1", "name": "f"}, {}, {"ok": True})])
    conversation = restore(conversation)
    history = adapter.convert_conversation_history_to_adapter_format(conversation, model="test")
    assert history[0]["content"] == blocks
    assert history[1]["content"][0]["tool_use_id"] == "tool_1"
    assert conversation.messages[0].provider_data["container"] == {"id": "container_1"}
    foreign = Conversation([Message("assistant", "text", thinking_responses=[ThinkingResponse("secret", "foreign_sig")])])
    assert "foreign_sig" not in json.dumps(adapter.convert_conversation_history_to_adapter_format(foreign))


def test_claude_stream_retains_redaction_signatures_json_and_citations():
    processor = ClaudeStreamProcessor()
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "opaque"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": "reason"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'}},
        {"type": "content_block_stop", "index": 2},
    ]
    for event in events:
        processor.process_event(obj(event))
    assert processor.content_blocks == [
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "thinking": "reason", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}},
    ]


def test_mistral_chunked_response_is_display_text_with_exact_native_replay():
    from mistralai.client.models import AssistantMessage
    native = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
        {"type": "text", "text": "answer"},
    ]}
    response = NS(id="r1", choices=[NS(message=obj(native))], usage=None)
    adapter = MistralAdapter()
    message = adapter._message_from_response("test", response)
    assert message.content == "answer"
    assert message.thinking_responses[0].content == "reason"
    history, _ = adapter.convert_conversation_history_to_adapter_format(restore(Conversation([message])), "test")
    assert history[1] == native
    AssistantMessage.model_validate(history[1])


def test_edited_native_message_uses_edited_text_instead_of_stale_response():
    message = Message("assistant", "original", provider="openai", model="test",
                      provider_data={"output": [{"type": "message", "content": []}]})
    message.content = "edited"
    restored = restore(Conversation([message]))
    payload = make_openai().convert_conversation_history_to_adapter_format(restored.messages, "test")
    assert payload[0]["content"][0]["text"] == "edited"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("store", [False, True])
def test_openai_tool_loop_replays_caller_and_distinct_call_id(asynchronous, store):
    adapter = make_openai()
    native = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
              {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
               "arguments": '{ "x": 1 }', "caller": {"type": "program", "caller_id": "prog_1"}}]
    responses = [openai_response(output=obj(native)), openai_response("resp_2")]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    if asynchronous:
        adapter._async_client.responses.create = create
    else:
        adapter._client.responses.create = create
    conversation = Conversation([Message("user", "go")])
    calls = []
    def lookup(x: int):
        calls.append(x)
        return {"result": x}
    if asynchronous:
        asyncio.run(adapter.request_llm_async("test", conversation, functions=[lookup], additional_parameters={"store": store}))
    else:
        adapter.request_llm("test", conversation, functions=[lookup], additional_parameters={"store": store})
    assert calls == [1]
    params = create.call_args.kwargs
    result = params["input"][-1]
    assert result["call_id"] == "call_1"
    assert result["caller"] == native[1]["caller"]
    if store:
        assert params["previous_response_id"] == "resp_1"
        assert len(params["input"]) == 1
    else:
        assert "previous_response_id" not in params
        assert params["input"][1:3] == native
        assert not conversation.continuations
    replay = adapter.convert_conversation_history_to_adapter_format(restore(conversation).messages, "test")
    assert replay[1:3] == native
    assert replay[3] == result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("agent_type", [None, "antigravity"])
def test_google_tool_round_then_expired_continuation_preserves_full_history(asynchronous, agent_type):
    adapter = make_google(agent_type)
    steps = [{"type": "thought", "signature": "signed", "summary": []},
             {"type": "function_call", "id": "call_1", "name": "lookup", "arguments": {"x": 1}}]
    responses = [google_response("i1", obj(steps), "env_1"),
                 google_response("i2", environment_id="env_1"),
                 MissingState("Interaction i2 not found"),
                 google_response("i3", environment_id="env_1")]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    if asynchronous:
        adapter._client.aio.interactions.create = create
    else:
        adapter._client.interactions.create = create
    calls = []
    def lookup(x: int):
        calls.append(x)
        return {"value": x}
    conversation = Conversation([Message("user", "go")])
    def request():
        if asynchronous:
            return asyncio.run(adapter.request_llm_async("test", conversation, functions=[lookup]))
        return adapter.request_llm("test", conversation, functions=[lookup])
    request()
    assert calls == [1]
    followup = create.call_args_list[1].kwargs
    assert followup["previous_interaction_id"] == "i1"
    assert followup["input"][0]["call_id"] == "call_1"
    assert followup["input"][0]["type"] == "function_result"
    conversation = restore(conversation)
    conversation.messages.extend([Message("user", "new 1"), Message("user", "new 2")])
    request()
    replay = create.call_args.kwargs
    assert "previous_interaction_id" not in replay
    assert replay["input"][1:3] == steps
    assert replay["input"][3] == followup["input"][0]
    assert calls == [1]  # Replay must not re-execute previous client tools.
    if agent_type:
        assert replay["environment"] == "env_1"


@pytest.mark.parametrize("provider", ["openai", "google"])
def test_unrelated_404_is_not_retried(provider):
    conversation = Conversation([Message("assistant", "answer")])
    conversation.checkpoint(provider, "test", "state_1")
    conversation.messages.append(Message("user", "question"))
    if provider == "openai":
        adapter = make_openai()
        create = adapter._client.responses.create
    else:
        adapter = make_google()
        create = adapter._client.interactions.create
    create.side_effect = MissingState("Model not found")
    with pytest.raises(MissingState):
        adapter.request_llm("test", conversation)
    assert create.call_count == 1
    assert conversation.continuation(provider, "test")["response_id"] == "state_1"


def test_grok_sdk_native_replay_preserves_encrypted_content_and_tool_outputs():
    from xai_sdk.chat import BaseChat, Response
    from xai_sdk.proto import chat_pb2
    from llm_platform.adapters.grok_adapter import GrokAdapter

    proto = chat_pb2.GetChatCompletionResponse(id="r1")
    output = proto.outputs.add()
    output.message.role = chat_pb2.ROLE_ASSISTANT
    output.message.content = "answer"
    output.message.encrypted_content = "encrypted"
    output.message.reasoning_content = "reason"
    call = output.message.tool_calls.add()
    call.id = "call_1"
    call.function.name = "lookup"
    call.function.arguments = '{ "x": 1 }'
    response = Response(proto, None)
    expected = BaseChat(None, None, None)
    expected.append(response)
    adapter = GrokAdapter()
    message = adapter._build_message_from_response(response, "test")
    replay = BaseChat(None, None, None)
    adapter.convert_conversation_history_to_adapter_format(replay, restore(Conversation([message])), "test")
    assert list(replay.messages) == list(expected.messages)
    assert replay.messages[0].encrypted_content == "encrypted"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_claude_request_replays_native_tool_exchange_and_container(asynchronous):
    adapter = AnthropicAdapter()
    adapter.correct_max_tokens = lambda model, history, maximum: maximum
    adapter.correct_max_tokens_async = AsyncMock(side_effect=lambda model, history, maximum: maximum)
    blocks = [{"type": "thinking", "thinking": "reason", "signature": "sig"},
              {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]
    def response(content, stop_reason):
        return NS(id="msg_1", model="provider-canonical-model", content=obj(content), stop_reason=stop_reason,
                  container=NS(id="container_1"), usage=NS(input_tokens=1, output_tokens=2))
    responses = [response(blocks, "tool_use"), response([{"type": "text", "text": "answer"}], "end_turn")]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    if asynchronous:
        adapter._async_client.beta.messages.create = create
    else:
        adapter._client.beta.messages.create = create
    def lookup():
        return {"answer": 42}
    conversation = Conversation([Message("user", "go")], "system")
    if asynchronous:
        asyncio.run(adapter.request_llm_async("test", conversation, functions=[lookup]))
    else:
        adapter.request_llm("test", conversation, functions=[lookup])
    params = create.call_args.kwargs
    assert params["messages"][1]["content"] == blocks
    assert params["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    assert params["container"] == "container_1"
    assert all(m.model == "test" for m in conversation.messages if m.role == "assistant")


def test_structured_openai_replay_omits_sdk_parsed_fields_only():
    from llm_platform.adapters.serializers import openai_output_to_input
    items = [{"type": "message", "content": [
        {"type": "output_text", "text": '{"parsed":1}', "parsed": {"parsed": 1}}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
         "arguments": '{"parsed_arguments":1}', "parsed_arguments": {"parsed_arguments": 1}}]
    replay = openai_output_to_input(items)
    assert "parsed" not in replay[0]["content"][0]
    assert "parsed_arguments" not in replay[1]
    assert replay[1]["arguments"] == items[1]["arguments"]
    assert "parsed" in items[0]["content"][0]


@pytest.mark.parametrize("arguments", [{"x": 1}, '{ "x": 1 }'])
def test_foreign_tool_history_has_valid_argument_types_and_matching_ids(arguments):
    from llm_platform.adapters.kimi_adapter import KimiAdapter
    from llm_platform.adapters.grok_adapter import GrokAdapter
    from xai_sdk.chat import BaseChat
    conversation = restore(Conversation([
        Message("assistant", "", function_calls=[FunctionCall("item_1", "f", arguments, "call_1")]),
        Message("function", "", function_responses=[FunctionResponse("f", {"ok": True}, call_id="call_1")]),
    ]))
    openai = make_openai().convert_conversation_history_to_adapter_format(conversation.messages, "test")
    assert json.loads(openai[0]["arguments"]) == {"x": 1}
    assert openai[0]["call_id"] == openai[1]["call_id"] == "call_1"
    google = make_google()._build_input_from_conversation(conversation, "test")
    assert google[0]["arguments"] == {"x": 1}
    assert google[0]["id"] == google[1]["call_id"] == "call_1"
    claude = AnthropicAdapter().convert_conversation_history_to_adapter_format(conversation, model="test")
    call = next(block for block in claude[0]["content"] if block["type"] == "tool_use")
    assert call["input"] == {"x": 1}
    assert call["id"] == claude[1]["content"][0]["tool_use_id"] == "call_1"
    for adapter in (OpenRouterAdapter(), KimiAdapter(), MistralAdapter()):
        history, _ = adapter.convert_conversation_history_to_adapter_format(conversation, "test")
        assert json.loads(history[1]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
        assert history[1]["tool_calls"][0]["id"] == history[2]["tool_call_id"] == "call_1"
    chat = BaseChat(None, None, None)
    GrokAdapter().convert_conversation_history_to_adapter_format(chat, conversation, "test")
    assert json.loads(chat.messages[0].tool_calls[0].function.arguments) == {"x": 1}
    assert chat.messages[0].tool_calls[0].id == chat.messages[1].tool_call_id == "call_1"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_deep_research_resumes_all_unsent_turns(asynchronous):
    adapter = make_google("deep_research")
    adapter.model_config["test"]["background_mode"] = True
    steps = obj([{"type": "model_output", "content": [{"type": "text", "text": "report"}]}])
    responses = [google_response("i1", steps), google_response("i2", steps)]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    if asynchronous:
        adapter._client.aio.interactions.create = create
    else:
        adapter._client.interactions.create = create
    conversation = Conversation([Message("user", "first")], "system")
    def request():
        options = {"agent_config": {"thinking_summaries": "auto"}}
        if asynchronous:
            return asyncio.run(adapter.request_llm_async("test", conversation, additional_parameters=options))
        return adapter.request_llm("test", conversation, additional_parameters=options)
    request()
    conversation = restore(conversation)
    conversation.messages.extend([Message("user", "second"), Message("user", "third")])
    request()
    params = create.call_args.kwargs
    assert params["previous_interaction_id"] == "i1"
    assert [item["content"][0]["text"] for item in params["input"]] == ["System instructions:\nsystem", "second", "third"]
    assert params["agent_config"]["type"] == "deep-research"
    assert "system_instruction" not in params
    replay = adapter._replay_interaction_kwargs(conversation, "test", params)
    assert "previous_interaction_id" not in replay
    assert replay["input"][0]["content"][0]["text"] == "System instructions:\nsystem"
    assert replay["input"][1]["content"][0]["text"] == "first"
