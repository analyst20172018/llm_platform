from typing import Any, Dict

from llm_platform.types import AdditionalParameters

from .openai_compatible_adapter import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """DeepSeek adapter (OpenAI-compatible Chat Completions API).

    DeepSeek V4 reasons before answering and returns the chain of thought in
    `reasoning_content`, which the shared base captures as a `ThinkingResponse`.
    Two knobs control it: `reasoning_effort`, a regular OpenAI SDK argument, and
    the thinking toggle, which is not part of the SDK signature and therefore has
    to travel inside `extra_body`.
    """

    BASE_URL = "https://api.deepseek.com"
    ENV_VAR = "DEEPSEEK_API_KEY"
    SUPPORTS_TOOLS = True
    JSON_OUTPUT_MODE = "json_object"

    # Thinking mode silently ignores these, so they are dropped instead of sent.
    SAMPLING_PARAMETERS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")

    def _build_request_params(
        self, model: str, additional_parameters: AdditionalParameters
    ) -> Dict[str, Any]:
        request_params = super()._build_request_params(model, additional_parameters)

        thinking = request_params.pop("thinking", None)
        if thinking is None:
            return request_params

        request_params.setdefault("extra_body", {})["thinking"] = thinking

        if thinking.get("type") != "disabled":
            for parameter in self.SAMPLING_PARAMETERS:
                request_params.pop(parameter, None)

        return request_params
