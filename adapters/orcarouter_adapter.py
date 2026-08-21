from .openai_compatible_adapter import OpenAICompatibleAdapter


class OrcarouterAdapter(OpenAICompatibleAdapter):
    """OrcaRouter adapter (OpenAI-compatible Chat Completions API)."""

    BASE_URL = "https://api.orcarouter.ai/v1"
    ENV_VAR = "ORCAROUTER_API_KEY"
