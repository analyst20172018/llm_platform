from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from llm_platform.adapters.orcarouter_adapter import OrcarouterAdapter
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation, Message


MODEL = "obsidian/Qwen3.8-27B"


def test_orcarouter_is_available_but_has_no_registered_model():
    handler = APIHandler()

    assert handler.model_config[MODEL] is None
    with pytest.raises(ValueError, match="not defined"):
        handler.get_adapter(MODEL)
    adapter = handler._lazy_initialization_of_adapter("OrcarouterAdapter")

    assert isinstance(adapter, OrcarouterAdapter)
    assert adapter._client is None
    assert adapter.BASE_URL == "https://api.orcarouter.ai/v1"
    assert adapter.ENV_VAR == "ORCAROUTER_API_KEY"


def test_orcarouter_request_uses_openai_compatible_chat_completions():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer."))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    adapter = OrcarouterAdapter()
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = response
    conversation = Conversation(
        messages=[Message(role="user", content="Hello")],
        system_prompt="Be helpful.",
    )

    message = adapter.request_llm(
        MODEL,
        conversation,
        additional_parameters={"max_tokens": 123},
    )

    adapter._client.chat.completions.create.assert_called_once_with(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=123,
    )
    assert message.content == "Answer."
    assert message.usage == {
        "model": MODEL,
        "completion_tokens": 5,
        "prompt_tokens": 10,
    }
    assert conversation.messages[-1] is message
