from .openai_compatible_adapter import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """DeepSeek adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://api.deepseek.com"
    ENV_VAR = "DEEPSEEK_API_KEY"
