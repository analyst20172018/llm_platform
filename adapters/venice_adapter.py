"""Venice chat completions through the existing OpenAI SDK."""

from llm_platform.services.files import AudioFile, ImageFile, VideoFile

from .openai_compatible_adapter import OpenAICompatibleAdapter


class VeniceAdapter(OpenAICompatibleAdapter):
    BASE_URL = "https://api.venice.ai/api/v1"
    ENV_VAR = "VENICE_API_KEY"
    SUPPORTS_TOOLS = True
    JSON_OUTPUT_MODE = "json_schema"

    def _api_model_name(self, model: str) -> str:
        model_config = self.model_config[model]
        return (model_config["api_model"] if model_config else None) or model

    def _build_request_params(self, model, additional_parameters):
        parameters = super()._build_request_params(model, additional_parameters)
        venice_parameters = parameters.pop("venice_parameters", {})
        # Use the caller's system prompt without Venice's additional default prompt.
        venice_parameters.setdefault("include_venice_system_prompt", False)
        parameters.setdefault("extra_body", {})["venice_parameters"] = venice_parameters
        if parameters.get("response_format"):
            # Venice structured outputs are incompatible with parallel tool calls.
            parameters["parallel_tool_calls"] = False
        return parameters

    def convert_conversation_history_to_adapter_format(self, the_conversation, model, **kwargs):
        model_config = self.model_config[model]
        inputs = model_config.inputs if model_config else ["text"]
        for message in the_conversation.messages:
            for file in message.files or []:
                for file_type, modality in (
                    (ImageFile, "image"), (VideoFile, "video"), (AudioFile, "audio")
                ):
                    if isinstance(file, file_type) and modality not in inputs:
                        raise ValueError(f"Model {model} does not support {modality} input.")
        return super().convert_conversation_history_to_adapter_format(
            the_conversation, model, **kwargs
        )

    def _video_content(self, file, model):
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{file.mime_type};base64,{file.base64}"},
        }
