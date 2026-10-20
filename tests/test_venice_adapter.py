"""Venice routing and real SDK serialization, with all HTTP calls intercepted."""

import asyncio
from copy import deepcopy
import json

import httpx
from openai import AsyncOpenAI, OpenAI
from PIL import Image
import pytest

from llm_platform.adapters.venice_adapter import VeniceAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation
from llm_platform.services.files import AudioFile, ImageFile, TextDocumentFile, VideoFile


ABLITERATED = "abliteration-abliterated-model-large-v2"
ALIAS = "abliterated-model-large-v2"
QWEN = "qwen-3-8-27b"
MODELS = [ABLITERATED, ALIAS, QWEN]


def completion(model, *, tool_call=False):
    message = {"role": "assistant", "content": '{"answer": 4}', "reasoning_content": "Checked."}
    if tool_call:
        message.update(content=None, tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "double", "arguments": '{"value": 2}'},
        }])
    return {
        "id": "chatcmpl_test", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_call else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_catalog_and_lazy_clients(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "venice-test-key")
    handler = APIHandler()
    adapter = handler.get_adapter(ALIAS)
    assert isinstance(adapter, VeniceAdapter)
    assert adapter._client is None
    assert adapter._async_client is None
    assert all(handler.get_adapter(model) is adapter for model in MODELS)
    assert handler.model_config[ABLITERATED].context_window == 1_000_000
    assert handler.model_config[QWEN].context_window == 262_144
    assert handler.model_config[ALIAS].visible is False
    assert handler.model_config[QWEN].inputs == ["text", "image", "video"]
    assert handler.model_config[QWEN].reasoning_effort == ["none", "low", "medium", "xhigh"]
    # Venice's live capability flag says effort is unsupported for Abliterated.
    assert handler.model_config[ABLITERATED].reasoning_effort == []
    for client in (adapter.client, adapter.async_client):
        assert str(client.base_url) == "https://api.venice.ai/api/v1/"
        assert client.api_key == "venice-test-key"
    adapter.client.close()
    asyncio.run(adapter.async_client.close())


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_tool_loop_schema_reasoning_and_alias(model, asynchronous):
    requests = []
    api_model = ABLITERATED if model == ALIAS else model

    def respond(request):
        assert str(request.url) == "https://api.venice.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(api_model, tool_call=len(requests) == 1))

    handler = APIHandler(system_prompt="Be helpful.")
    adapter = handler.get_adapter(model)
    schema = {
        "type": "object", "properties": {"answer": {"type": "integer"}},
        "required": ["answer"], "additionalProperties": False,
    }
    parameters = {
        "max_tokens": 128, "structured_output": schema, "tool_choice": "required",
        "temperature": 0.3, "top_p": 0.9,
        "venice_parameters": {"include_venice_system_prompt": True},
    }
    if model == QWEN:
        parameters["reasoning_effort"] = "low"
    original = deepcopy(parameters)
    tool_results = []

    def double(value: int):
        tool_results.append(value)
        return value * 2

    async def double_async(value: int):
        return double(value)

    double_async.__name__ = "double"
    transport = httpx.MockTransport(respond)
    if asynchronous:
        async def run():
            async with AsyncOpenAI(
                api_key="test", base_url=adapter.BASE_URL,
                http_client=httpx.AsyncClient(transport=transport),
            ) as client:
                adapter._async_client = client
                return await handler.request_async(
                    model, "Double 2.", functions=[double_async], additional_parameters=parameters
                )
        result = asyncio.run(run())
    else:
        with OpenAI(
            api_key="test", base_url=adapter.BASE_URL,
            http_client=httpx.Client(transport=transport),
        ) as client:
            adapter._client = client
            result = handler.request(
                model, "Double 2.", functions=[double], additional_parameters=parameters
            )

    assert parameters == original
    assert tool_results == [2]
    assert len(requests) == 2
    first, second = requests
    for payload in requests:
        assert payload["model"] == api_model
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.9
        assert payload["venice_parameters"] == {"include_venice_system_prompt": True}
        assert "extra_body" not in payload
        assert "structured_output" not in payload
        assert payload["response_format"]["json_schema"]["schema"] == schema
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["parallel_tool_calls"] is False
        if model == QWEN:
            assert payload["reasoning_effort"] == "low"
    assert first["tool_choice"] == "required"
    assert second["tool_choice"] == "auto"
    assert second["messages"][-2]["reasoning_content"] == "Checked."
    assert second["messages"][-2]["tool_calls"][0]["id"] == "call_1"
    assert second["messages"][-1]["tool_call_id"] == "call_1"
    assert result.provider == "venice"
    assert result.model == model
    assert result.status == "completed"
    assert result.content == '{"answer": 4}'
    assert result.thinking_responses[0].content == "Checked."
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 5
    restored = Conversation.read_from_json(json.loads(json.dumps(handler.the_conversation.save_to_json())))
    history, _ = adapter.convert_conversation_history_to_adapter_format(restored, model)
    assert history[-1]["reasoning_content"] == "Checked."


@pytest.mark.parametrize("model,limit", [(ABLITERATED, 32768), (ALIAS, 32768), (QWEN, 65536)])
def test_defaults_and_supported_files_on_wire(model, limit):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(model))

    handler = APIHandler()
    adapter = handler.get_adapter(model)
    files = [TextDocumentFile("Document text.", "notes.txt")]
    if model == QWEN:
        files += [
            ImageFile.from_pil_image(Image.new("RGB", (1, 1)), "image.png"),
            VideoFile(b"video", "clip.mp4"),
        ]
    with OpenAI(
        api_key="test", base_url=adapter.BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        adapter._client = client
        handler.request(model, "Describe.", files=files)
    payload = requests[0]
    assert payload["max_tokens"] == limit
    assert payload["venice_parameters"] == {"include_venice_system_prompt": False}
    assert "reasoning_effort" not in payload
    assert "response_format" not in payload
    content = payload["messages"][-1]["content"]
    assert "Document text." in content[0]["text"]
    if model == QWEN:
        assert content[-2]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[-1] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlkZW8="}}


@pytest.mark.parametrize("model,file,modality", [
    (ALIAS, ImageFile(b"image", "image.png"), "image"),
    (ABLITERATED, VideoFile(b"video", "clip.mp4"), "video"),
    (ABLITERATED, AudioFile(b"audio", "audio.mp3"), "audio"),
    (QWEN, AudioFile(b"audio", "audio.mp3"), "audio"),
])
def test_unsupported_media_fails_before_client_creation(model, file, modality):
    handler = APIHandler()
    with pytest.raises(ValueError, match=f"does not support {modality}"):
        handler.request(model, "Describe.", files=[file])
    assert handler.get_adapter(model)._client is None
