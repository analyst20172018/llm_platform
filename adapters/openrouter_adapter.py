from typing import Any, Dict

from llm_platform.types import AdditionalParameters

from .openai_compatible_adapter import OpenAICompatibleAdapter


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_VAR = "OPENROUTER_API_KEY"
    SUPPORTS_TOOLS = True
    JSON_OUTPUT_MODE = "json_schema"

    def _build_request_params(
        self, model: str, additional_parameters: AdditionalParameters
    ) -> Dict[str, Any]:
        request_params = super()._build_request_params(model, additional_parameters)

        if reasoning := additional_parameters.get("reasoning"):
            request_params.setdefault("extra_body", {})["reasoning"] = reasoning

        if additional_parameters.get("structured_output"):
            request_params.setdefault("extra_body", {}).setdefault("provider", {})["require_parameters"] = True

        return request_params

    def _prepare_tools(self, model, functions, request_params):
        tool_map = super()._prepare_tools(model, functions, request_params)
        if tool_map:
            # Route only to endpoints that honor all requested parameters.
            request_params.setdefault("extra_body", {}).setdefault("provider", {})["require_parameters"] = True
        return tool_map
