"""Offline regressions for the capability-alignment API audit fixes."""

import asyncio
import base64
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel

from llm_platform.adapters.deepseek_adapter import DeepSeekAdapter
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.mistral_adapter import MistralAdapter
from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.adapters.zai_adapter import ZaiAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import ImageFile, PDFDocumentFile, VideoFile
from llm_platform.tools.base import BaseTool


PROVIDERS = [(DeepSeekAdapter, "deepseek-v4-pro"), (OpenRouterAdapter, "meta/muse-spark-1.3")]


class Answer(BaseModel):
    value: int


class Add(BaseTool):
    """Add two numbers."""

    class InputModel(BaseModel):
        value: int
        amount: int = 1

    def __call__(self, value: int, amount: int = 1):
        return value + amount


def conversation(files=None):
    return Conversation(system_prompt="Be helpful.", messages=[Message(role="user", content="Answer.", files=files)])


def response(*, calls=None, finish="stop", **native):
    return NS(id="r1", usage=None, choices=[NS(finish_reason=finish, message=NS(
        role="assistant", content='{"value": 2}', tool_calls=calls or [], **native
    ))])


def tool_call(name="Add", arguments='{"value": 1}', call_id="call_1"):
    return NS(id=call_id, type="function", function=NS(name=name, arguments=arguments))


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash", "kimi-k3"])
def test_effort_choices_reach_the_provider(model):
    handler = APIHandler()
    assert handler.model_config[model].get_parameter_options("reasoning_effort") == ["low", "high", "max"]
    for effort in ("low", "high", "max"):
        params = handler._prepare_additional_parameters(model, {"reasoning_effort": effort})
        assert handler.get_adapter(model)._build_request_params(model, params)["reasoning_effort"] == effort
    assert handler.model_config["grok-4.6"].context_window == 500_000


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp",
                                   "meta/muse-spark-1.3", "mistral-large-latest", "glm-5.3"])
def test_json_and_tool_capabilities_are_exposed(model):
    handler = APIHandler()
    assert handler.model_config[model].get_parameter("function_calling")
    assert handler._prepare_additional_parameters(model, {"structured_output": Answer})["structured_output"] is Answer


@pytest.mark.parametrize("adapter_cls,model", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_loop_preserves_reasoning_ids_defaults_and_forced_choice(adapter_cls, model, asynchronous):
    adapter = adapter_cls()
    native = {"reasoning_content": "think"} if adapter_cls is DeepSeekAdapter else {
        "reasoning": "think", "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}]
    }
    responses = [response(calls=[tool_call()], finish="tool_calls", **native), response()]
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    client = MagicMock()
    client.chat.completions.create = create
    if asynchronous:
        adapter._async_client = client
    else:
        adapter._client = client
    state = conversation()
    params = {"tool_choice": "required", "structured_output": Answer, "extra_body": {"custom": 1}}
    original = copy.deepcopy(params)
    callback = MagicMock()
    kwargs = dict(functions=[Add()], additional_parameters=params, tool_output_callback=callback)
    result = asyncio.run(adapter.request_llm_async(model, state, **kwargs)) if asynchronous else adapter.request_llm(model, state, **kwargs)
    first, second = [call.kwargs for call in create.call_args_list]
    assert first["tool_choice"] == "required" and second["tool_choice"] == "auto"
    assistant, tool = second["messages"][-2:]
    for key, value in native.items():
        assert assistant[key] == value
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "call_1"
    assert json.loads(tool["content"]) == {"text": 2}
    callback.assert_called_once_with("Add", {"value": 1}, 2)
    assert params == original
    assert state.messages[-1] is result
    restored = Conversation.read_from_json(json.loads(json.dumps(state.save_to_json())))
    history, _ = adapter.convert_conversation_history_to_adapter_format(restored, model)
    assert history[-3:-1] == [assistant, tool]
    if adapter_cls is OpenRouterAdapter:
        assert first["extra_body"]["provider"] == {"require_parameters": True}
        assert first["response_format"]["type"] == "json_schema"
    else:
        assert first["response_format"] == {"type": "json_object"}
        assert '"value"' in first["messages"][0]["content"]


@pytest.mark.parametrize("adapter_cls,model", PROVIDERS)
def test_async_callable_tool_and_callback(adapter_cls, model):
    async def lookup(value: int = 3):
        return value

    adapter = adapter_cls()
    adapter._async_client = MagicMock()
    adapter._async_client.chat.completions.create = AsyncMock(side_effect=[
        response(calls=[tool_call("lookup", "{}")], finish="tool_calls"), response()
    ])
    callback = AsyncMock()
    asyncio.run(adapter.request_llm_async(model, conversation(), functions=[lookup], tool_output_callback=callback))
    callback.assert_awaited_once_with("lookup", {}, 3)


@pytest.mark.parametrize("adapter_cls,model", PROVIDERS)
@pytest.mark.parametrize("finish", ["length", "content_filter", "error"])
def test_terminal_response_does_not_execute_tools(adapter_cls, model, finish):
    adapter = adapter_cls()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = response(calls=[tool_call()], finish=finish)
    callback = MagicMock()
    adapter.request_llm(model, conversation(), functions=[Add()], tool_output_callback=callback)
    callback.assert_not_called()
    assert adapter._client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter_cls,model", PROVIDERS)
def test_tool_round_limit_preserves_completed_rounds(adapter_cls, model, asynchronous):
    adapter = adapter_cls()
    create = AsyncMock(return_value=response(calls=[tool_call()], finish="tool_calls")) if asynchronous else MagicMock(return_value=response(calls=[tool_call()], finish="tool_calls"))
    client = MagicMock()
    client.chat.completions.create = create
    adapter._client = adapter._async_client = client
    state = conversation()
    with patch("llm_platform.adapters.openai_compatible_adapter.MAX_TOOL_ROUNDS", 2), pytest.raises(RuntimeError, match="maximum tool-calling"):
        if asynchronous:
            asyncio.run(adapter.request_llm_async(model, state, functions=[Add()]))
        else:
            adapter.request_llm(model, state, functions=[Add()])
    assert create.call_count == 2
    assert len(state.messages) == 5


@pytest.mark.parametrize("value", [True, Answer, Answer.model_json_schema(), {"type": "json_schema", "json_schema": {"name": "answer", "schema": Answer.model_json_schema(), "strict": True}}])
@pytest.mark.parametrize("with_tools", [False, True])
def test_mistral_json_output_uses_sdk_response_format(value, with_tools):
    from mistralai.client.models import ResponseFormat

    adapter = MistralAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.complete.return_value = response()
    state = conversation()
    adapter.request_llm("mistral-large-latest", state, functions=[Add()] if with_tools else None,
                        additional_parameters={"structured_output": value})
    request = adapter._client.chat.complete.call_args.kwargs
    output_format = request["response_format"]
    ResponseFormat.model_validate(output_format)
    assert output_format["type"] == ("json_object" if value is True else "json_schema")
    assert "structured_output" not in request
    assert "JSON" in request["messages"][0]["content"]
    assert state.system_prompt == "Be helpful."


@pytest.mark.parametrize("file", [ImageFile(b"image", "image.png"), VideoFile(b"video", "clip.mp4")])
def test_glm_flash_visual_parts_and_text_only_guard(file):
    adapter = ZaiAdapter()
    history, _ = adapter.convert_conversation_history_to_adapter_format(conversation([file]), "glm-5.3-flash")
    key = "image_url" if isinstance(file, ImageFile) else "video_url"
    part = history[-1]["content"][-1]
    assert part["type"] == key
    assert part[key]["url"].endswith(file.base64)
    with pytest.raises(ValueError, match="does not support"):
        adapter.convert_conversation_history_to_adapter_format(conversation([file]), "glm-5.3")


def test_glm_flash_pdf_keeps_visual_bytes_and_rejects_mixed_native_parts():
    adapter = ZaiAdapter()
    pdf = PDFDocumentFile(b"pdf", "document.pdf")
    history, _ = adapter.convert_conversation_history_to_adapter_format(conversation([pdf]), "glm-5.3-flash")
    assert history[-1]["content"][0] == {"type": "file", "file": {
        "file_data": "data:application/pdf;base64,cGRm", "filename": "document.pdf"
    }}
    with pytest.raises(ValueError, match="cannot combine"):
        adapter.convert_conversation_history_to_adapter_format(conversation([pdf, ImageFile(b"image", "image.png")]), "glm-5.3-flash")


@pytest.mark.parametrize("pages,size,valid", [(1000, 50_000_000, True), (1001, 100, False), (1, 50_000_001, False)])
def test_gemini_pdf_boundaries(pages, size, valid):
    adapter = GoogleAdapter()
    pdf = PDFDocumentFile(b"pdf", "document.pdf")
    with patch.object(PDFDocumentFile, "number_of_pages", new_callable=PropertyMock, return_value=pages), patch.object(PDFDocumentFile, "size", new_callable=PropertyMock, return_value=size):
        if valid:
            assert adapter._convert_file_to_interaction_content(pdf)["type"] == "document"
        else:
            with pytest.raises(ValueError, match="50 MB / 1,000 page"):
                adapter._convert_file_to_interaction_content(pdf)


def test_gemini_total_pdf_pages():
    with patch.object(PDFDocumentFile, "number_of_pages", new_callable=PropertyMock, return_value=600):
        with pytest.raises(ValueError, match="1,000 PDF pages"):
            GoogleAdapter()._build_input_from_conversation(conversation([PDFDocumentFile(b"a"), PDFDocumentFile(b"b")]))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_gemini_uploads_for_aggregate_encoded_size_without_mutating_history(asynchronous):
    from google.genai._gaos.types.interactions.documentcontent import DocumentContent

    adapter = GoogleAdapter()
    adapter._client = MagicMock()
    # Each PDF fits alone; the base64-encoded pair plus prompt does not.
    adapter.INLINE_REQUEST_MAX_BYTES = 3000
    content = [{"type": "text", "text": "Prompt"}] + [
        {"type": "document", "mime_type": "application/pdf", "data": base64.b64encode(b"x" * 1200).decode()}
        for _ in range(2)
    ]
    kwargs = {"model": "gemini-3.5-flash", "input": [{"type": "user_input", "content": content}]}
    original = copy.deepcopy(kwargs)
    def upload(**request):
        assert request["file"].read() == b"x" * 1200
        return NS(uri="https://example.test/files/pdf")
    if asynchronous:
        adapter._client.aio.files.upload = AsyncMock(side_effect=upload)
        adapter._client.aio.interactions.create = AsyncMock(return_value=NS(id="result"))
        asyncio.run(adapter._create_interaction_async(conversation(), "gemini-3.5-flash", **kwargs))
        request = adapter._client.aio.interactions.create.call_args.kwargs
    else:
        adapter._client.files.upload.side_effect = upload
        adapter._create_interaction(conversation(), "gemini-3.5-flash", **kwargs)
        request = adapter._client.interactions.create.call_args.kwargs
    assert kwargs == original
    parts = request["input"][0]["content"][1:]
    assert any("uri" in part for part in parts)
    for part in parts:
        DocumentContent.model_validate(part)
    assert len(json.dumps(request).encode()) < adapter.INLINE_REQUEST_MAX_BYTES
