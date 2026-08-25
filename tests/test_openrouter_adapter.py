from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.core.llm_handler import APIHandler


MODEL = "stealth/ox-alpha"


def test_reasoning_is_enabled_by_default_for_ox_alpha():
    handler = APIHandler()

    assert isinstance(handler.get_adapter(MODEL), OpenRouterAdapter)
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "reasoning": {"enabled": True},
        "max_tokens": 131_072,
    }


def test_reasoning_is_sent_in_openrouter_extra_body():
    adapter = OpenRouterAdapter()

    request_params = adapter._build_request_params(
        MODEL,
        {
            "max_tokens": 123,
            "reasoning": {"enabled": True},
        },
    )

    assert request_params == {
        "max_tokens": 123,
        "extra_body": {"reasoning": {"enabled": True}},
    }
