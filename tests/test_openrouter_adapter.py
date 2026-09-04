from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.core.llm_handler import APIHandler


MODEL = "meta/muse-spark-1.3"


def test_reasoning_effort_defaults_for_muse_spark():
    handler = APIHandler()

    assert isinstance(handler.get_adapter(MODEL), OpenRouterAdapter)
    assert handler._prepare_additional_parameters(MODEL, None) == {
        "reasoning": {"effort": "medium"},
        "max_tokens": 943_718,
    }


def test_reasoning_is_sent_in_openrouter_extra_body():
    adapter = OpenRouterAdapter()

    request_params = adapter._build_request_params(
        MODEL,
        {
            "max_tokens": 123,
            "reasoning": {"effort": "high"},
        },
    )

    assert request_params == {
        "max_tokens": 123,
        "extra_body": {"reasoning": {"effort": "high"}},
    }
