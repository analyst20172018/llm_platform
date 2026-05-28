from .openai_compatible_adapter import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """DeepSeek adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://api.deepseek.com"
    ENV_VAR = "DEEPSEEK_API_KEY"

    def _suppress_temperature(self, model: str) -> bool:
        # The deepseek-reasoner model does not accept a temperature parameter.
        return model == "deepseek-reasoner"
