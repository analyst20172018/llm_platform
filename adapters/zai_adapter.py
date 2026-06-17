import os

from .openai_compatible_adapter import OpenAICompatibleAdapter


class ZaiAdapter(OpenAICompatibleAdapter):
    """Z.AI adapter (GLM models) built on the official ``zai-sdk`` ``ZaiClient``.

    ``ZaiClient`` exposes the same OpenAI-compatible ``chat.completions.create``
    surface as the shared base, so only client construction differs.
    """

    BASE_URL = "https://api.z.ai/api/paas/v4/"
    ENV_VAR = "ZAI_API_KEY"

    def _build_client(self):
        from zai import ZaiClient
        return ZaiClient(api_key=os.getenv(self.ENV_VAR), base_url=self.BASE_URL)
