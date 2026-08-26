from llm_platform.adapters.zai_adapter import ZaiAdapter
from llm_platform.core.llm_handler import APIHandler


MODEL = "glm-5.3-flash"


def test_glm_flash_reasoning_defaults_are_normalized_from_model_config():
    handler = APIHandler()

    assert isinstance(handler.get_adapter(MODEL), ZaiAdapter)
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "max_tokens": 128000,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
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
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "web_search": False,
    }
    assert handler.get_adapter(MODEL)._build_request_params(MODEL, parameters) == {
        "max_tokens": 128000,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
