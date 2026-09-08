import hashlib
import hmac
from unittest.mock import MagicMock, call, patch

import pytest

from llm_platform.adapters.wiro_ai_adapter import WiroAIAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message


MODEL = "Qwen3.8-27B-Uncensored"


def response(data):
    result = MagicMock()
    result.json.return_value = data
    return result


def test_wiro_ai_model_is_registered_with_lazy_adapter():
    handler = APIHandler()

    adapter = handler.get_adapter(MODEL)
    model = handler.model_config[MODEL]

    assert isinstance(adapter, WiroAIAdapter)
    assert adapter._client is None
    assert model.context_window == 262_144
    assert model.inputs == ["text"]
    assert model["wiro_owner"] == "Qwen"
    assert model["wiro_model"] == MODEL
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "enableThinking": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "min_tokens": 500,
        "max_tokens": 0,
    }


def test_wiro_ai_signature_authentication(monkeypatch):
    monkeypatch.setenv("WIRO_API_KEY", "api-key")
    monkeypatch.setenv("WIRO_API_SECRET", "api-secret")

    with patch("llm_platform.adapters.wiro_ai_adapter.time.time_ns", return_value=123_000_000):
        headers = WiroAIAdapter._auth_headers()

    nonce = "123"
    expected_signature = hmac.new(
        b"api-key",
        f"api-secret{nonce}".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert headers == {
        "Content-Type": "application/json",
        "x-api-key": "api-key",
        "x-signature": expected_signature,
        "x-nonce": nonce,
    }


def test_wiro_ai_request_runs_polls_and_parses_structured_llm_output(monkeypatch):
    monkeypatch.setenv("WIRO_API_KEY", "api-key")
    monkeypatch.delenv("WIRO_API_SECRET", raising=False)
    adapter = WiroAIAdapter()
    adapter._client = MagicMock()
    adapter._client.post.side_effect = [
        response({"result": True, "taskid": "42", "socketaccesstoken": "token"}),
        response(
            {
                "result": True,
                "tasklist": [{"id": "42", "status": "task_start", "pexit": None}],
            }
        ),
        response(
            {
                "result": True,
                "tasklist": [
                    {
                        "id": "42",
                        "status": "task_postprocess_end",
                        "pexit": "0",
                        "totalcost": "0.0125",
                        "debugoutput": "merged output",
                        "outputs": [
                            {
                                "contenttype": "raw",
                                "content": {
                                    "raw": "raw output",
                                    "thinking": ["Reasoning step"],
                                    "answer": ["First answer.", "Second answer."],
                                },
                            }
                        ],
                    }
                ],
            }
        ),
    ]
    conversation = Conversation(
        messages=[Message(role="user", content="Hello")],
        system_prompt="Be helpful.",
    )

    with patch("llm_platform.adapters.wiro_ai_adapter.time.sleep") as sleep:
        message = adapter.request_llm(
            MODEL,
            conversation,
            additional_parameters={"enableThinking": True, "temperature": 0.7},
        )

    headers = {"Content-Type": "application/json", "x-api-key": "api-key"}
    assert adapter._client.post.call_args_list == [
        call(
            "https://api.wiro.ai/v1/Run/Qwen/Qwen3.8-27B-Uncensored",
            json={
                "enableThinking": "true",
                "temperature": 0.7,
                "prompt": "Hello",
                "system_prompt": "Be helpful.",
            },
            headers=headers,
            timeout=WiroAIAdapter.HTTP_TIMEOUT_SECONDS,
        ),
        call(
            "https://api.wiro.ai/v1/Task/Detail",
            json={"tasktoken": "token"},
            headers=headers,
            timeout=WiroAIAdapter.HTTP_TIMEOUT_SECONDS,
        ),
        call(
            "https://api.wiro.ai/v1/Task/Detail",
            json={"tasktoken": "token"},
            headers=headers,
            timeout=WiroAIAdapter.HTTP_TIMEOUT_SECONDS,
        ),
    ]
    sleep.assert_called_once_with(2)
    assert message.content == "First answer.\n\nSecond answer."
    assert [item.content for item in message.thinking_responses] == ["Reasoning step"]
    assert message.usage == {
        "model": MODEL,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "costs": 0.0125,
    }
    assert message.id == "42"
    assert conversation.messages[-1] is message


def test_wiro_ai_sends_complete_conversation_without_hidden_session_state():
    adapter = WiroAIAdapter()
    conversation = Conversation(
        messages=[
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Follow-up"),
        ]
    )

    assert adapter.convert_conversation_history_to_adapter_format(conversation) == (
        "User:\nFirst question\n\nAssistant:\nFirst answer\n\nUser:\nFollow-up"
    )


def test_wiro_ai_surfaces_task_failures(monkeypatch):
    monkeypatch.setenv("WIRO_API_KEY", "api-key")
    adapter = WiroAIAdapter()
    adapter._client = MagicMock()
    adapter._client.post.side_effect = [
        response({"result": True, "taskid": "42"}),
        response(
            {
                "result": True,
                "tasklist": [
                    {
                        "id": "42",
                        "status": "task_postprocess_end",
                        "pexit": "1",
                        "debugerror": "Model process failed",
                    }
                ],
            }
        ),
    ]

    with pytest.raises(RuntimeError, match="Model process failed"):
        adapter.request_llm(
            MODEL,
            Conversation(messages=[Message(role="user", content="Hello")]),
        )
