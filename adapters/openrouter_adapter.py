from typing import Any, Dict

from llm_platform.types import AdditionalParameters

from .openai_compatible_adapter import OpenAICompatibleAdapter


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_VAR = "OPENROUTER_API_KEY"

    def _build_request_params(
        self, model: str, additional_parameters: AdditionalParameters
    ) -> Dict[str, Any]:
        request_params = super()._build_request_params(model, additional_parameters)

        if reasoning := additional_parameters.get("reasoning"):
            request_params.setdefault("extra_body", {})["reasoning"] = reasoning

        return request_params
