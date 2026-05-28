from .openai_compatible_adapter import OpenAICompatibleAdapter


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_VAR = "OPENROUTER_API_KEY"
