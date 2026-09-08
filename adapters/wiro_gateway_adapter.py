"""Opt-in Wiro Direct LLM Chat route, separate from the Run/Task API."""

import os

from .openai_compatible_adapter import OpenAICompatibleAdapter
from .serializers import provider_dump


class WiroGatewayAdapter(OpenAICompatibleAdapter):
    BASE_URL = "https://llm.wiro.ai/v1"
    SUPPORTS_TOOLS = True
    JSON_OUTPUT_MODE = "json_schema"

    @staticmethod
    def _credential():
        key = os.getenv("WIRO_API_KEY")
        if not key:
            raise ValueError("WIRO_API_KEY is required for the Wiro gateway")
        secret = os.getenv("WIRO_API_SECRET")
        return f"{key}:{secret}" if secret else key

    def _build_client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.BASE_URL, api_key=self._credential())

    def _build_async_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=self.BASE_URL, api_key=self._credential())

    @staticmethod
    def _validate_contract(model, contract, functions, parameters):
        """Fail before generation if authenticated discovery cannot confirm support."""
        data = provider_dump(contract)
        capabilities = data.get("capabilities") or {}
        if data.get("id") != model or "chat" not in capabilities.get("endpoints", []):
            raise ValueError(f"Wiro does not advertise a Chat route for {model}")
        if "text" not in capabilities.get("input_modalities", []):
            raise ValueError(f"Wiro does not advertise text input for {model}")
        if functions and not capabilities.get("function_tools"):
            raise ValueError(f"Wiro does not advertise function tools for {model}")
        if parameters.get("structured_output"):
            mode = "json_object" if parameters["structured_output"] is True else "json_schema"
            if mode not in capabilities.get("structured_output_modes", []):
                raise ValueError(f"Wiro does not advertise {mode} for {model}")
            if mode == "json_schema" and not capabilities.get("strict_json_schema"):
                raise ValueError(f"Wiro does not advertise strict JSON schemas for {model}")
        maximum = (data.get("top_provider") or {}).get("max_completion_tokens")
        requested = parameters.get("max_completion_tokens", parameters.get("max_tokens"))
        if maximum is not None and requested is not None and requested > maximum:
            raise ValueError(f"max_tokens exceeds Wiro's advertised limit for {model}")

    def _build_request_params(self, model, additional_parameters):
        allowed = {"max_tokens", "max_completion_tokens", "tool_choice", "structured_output"}
        if additional_parameters.keys() - allowed:
            raise ValueError("Wiro gateway supports only token limits, local tools, and JSON output here")
        parameters = super()._build_request_params(model, additional_parameters)
        if "max_tokens" in parameters:
            parameters["max_completion_tokens"] = parameters.pop("max_tokens")
        return parameters

    def _prepare_history(self, conversation, model, structured_output=None):
        # This initial route is deliberately text-only. Documents are extracted by
        # the shared converter; enabling native media needs a separate contract check.
        from llm_platform.services.files import DocumentFile
        if any(not isinstance(file, DocumentFile)
               for message in conversation.messages for file in message.files or []):
            raise ValueError("The registered Wiro gateway route accepts text/documents only")
        return super()._prepare_history(conversation, model, structured_output)

    def request_llm(self, model, the_conversation, functions=None, tool_output_callback=None,
                    additional_parameters=None, **kwargs):
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        self._build_request_params(model, parameters)
        contract = self.client.models.retrieve(model)
        self._validate_contract(model, contract, functions, parameters)
        return super().request_llm(model, the_conversation, functions, tool_output_callback,
                                   additional_parameters=parameters)

    async def request_llm_async(self, model, the_conversation, functions=None,
                                tool_output_callback=None, additional_parameters=None, **kwargs):
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        self._build_request_params(model, parameters)
        contract = await self.async_client.models.retrieve(model)
        self._validate_contract(model, contract, functions, parameters)
        return await super().request_llm_async(
            model, the_conversation, functions, tool_output_callback,
            additional_parameters=parameters,
        )
