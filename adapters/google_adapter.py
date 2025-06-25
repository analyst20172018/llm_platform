import json
import logging
import os
import time
import uuid
from io import BytesIO
from typing import Callable, Dict, List, Tuple

from google import genai
from google.genai import types
from google.protobuf import struct_pb2

from .adapter_base import AdapterBase
from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, ExcelDocumentFile,
                                         ImageFile, MediaFile, PDFDocumentFile,
                                         TextDocumentFile, VideoFile)
from llm_platform.tools.base import BaseTool


class GoogleAdapter(AdapterBase):
    """
    Adapter for interacting with the Google Gemini API.
    """
    # Class-level constants for configuration and mapping
    GEMINI_ROLE_MAPPING = {'user': 'user', 'assistant': 'model'}
    REASONING_EFFORT_MAP = {'high': 24_576, 'medium': 8_000}
    IMAGEN_DEFAULT_MODEL = 'imagen-4.0-generate-preview-06-06'
    VEO_MODEL = 'veo-2.0-generate-001'

    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    def _convert_file_to_part(self, file: MediaFile) -> types.Part:
        """Converts a MediaFile object to a Gemini API Part."""
        if isinstance(file, ImageFile):
            return types.Part.from_bytes(data=file.file_bytes, mime_type=f"image/{file.extension}")
        if isinstance(file, AudioFile):
            return types.Part.from_bytes(data=file.file_bytes, mime_type="audio/mp3")
        if isinstance(file, (TextDocumentFile, ExcelDocumentFile)):
            text = f'<document name="{file.name}">{file.text}</document>'
            return types.Part.from_text(text=text)
        if isinstance(file, PDFDocumentFile):
            # Gemini has limits on direct PDF processing
            if file.size < 20_000_000 and file.number_of_pages < 3_600:
                return types.Part.from_bytes(data=file.bytes, mime_type="application/pdf")
            else:
                self.logger.warning(f"PDF '{file.name}' exceeds size/page limits; sending as text.")
                text = f'<document name="{file.name}">{file.text}</document>'
                return types.Part.from_text(text=text)
        raise TypeError(f"Unsupported file type for Gemini: {type(file).__name__}")

    def convert_conversation_history_to_adapter_format(self, conversation: Conversation) -> List[types.Content]:
        """
        Converts a Conversation object into the list of Content objects
        required by the Gemini API.
        """
        history = []
        for message in conversation.messages:
            try:
                role = self.GEMINI_ROLE_MAPPING[message.role]
            except KeyError:
                # 'function' role messages are generated from assistant responses, not directly mapped.
                if message.role == 'function':
                    continue
                raise ValueError(f"Invalid message role for Gemini: '{message.role}'")

            parts = []
            if message.content:
                parts.append(types.Part.from_text(text=message.content))

            if message.files:
                for file in message.files:
                    try:
                        parts.append(self._convert_file_to_part(file))
                    except TypeError as e:
                        self.logger.warning(e)

            if message.function_calls:
                for fc in message.function_calls:
                    struct_args = struct_pb2.Struct()
                    struct_args.update(json.loads(fc.arguments))
                    parts.append(types.Part.from_function_call(name=fc.name, args=struct_args))

            if parts:
                history.append(types.Content(role=role, parts=parts))

            # If an assistant message has function responses, add a subsequent 'function' role message
            if message.function_responses:
                response_parts = [
                    types.Part.from_function_response(name=fr.name, response=fr.response)
                    for fr in message.function_responses
                ]
                if response_parts:
                    history.append(types.Content(role="function", parts=response_parts))
        return history

    def _prepare_generation_config(self, model: str, the_conversation: Conversation, temperature: float,
                                   tools: List, additional_parameters: Dict, **kwargs) -> types.GenerateContentConfig:
        """Prepares the GenerateContentConfig for a Gemini API call."""
        # Gemini uses 'max_output_tokens'
        if 'max_tokens' in kwargs:
            kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
        else:
            kwargs['max_output_tokens'] = the_conversation.model_config.get_max_tokens(model)

        config_params = {
            "temperature": temperature,
            "tools": tools,
            "safety_settings": self.safety_settings,
        }

        if the_conversation.system_prompt:
            config_params["system_instruction"] = the_conversation.system_prompt

        reasoning_effort = kwargs.pop('reasoning', {}).get('effort', 'low')
        if thinking_budget := self.REASONING_EFFORT_MAP.get(reasoning_effort):
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=True
            )

        if "response_modalities" in additional_parameters:
            config_params["response_modalities"] = additional_parameters["response_modalities"]

        if additional_parameters.get("web_search"):
            config_params["tools"].append(types.Tool(google_search=types.GoogleSearchRetrieval()))
        if additional_parameters.get("url_context"):
            config_params["tools"].append(types.Tool(url_context=types.UrlContext()))

        config_params.update(kwargs)

        return types.GenerateContentConfig(**config_params)

    def _parse_gemini_response(self, response_candidate: types.Candidate, model_name: str,
                               usage_metadata: types.UsageMetadata) -> Message:
        """Parses a Gemini API response candidate into a Message object."""
        text_content, thoughts, files, function_calls = "", [], [], []

        for part in response_candidate.content.parts:
            if fc := part.function_call:
                function_calls.append(FunctionCall(
                    id=str(uuid.uuid4()),
                    name=fc.name,
                    arguments=json.dumps(dict(fc.args))
                ))
            elif part.text:
                if part.thought:
                    thoughts.append(ThinkingResponse(content=part.text, id=None))
                else:
                    text_content += part.text
            elif part.inline_data and part.inline_data.mime_type == "image/png":
                files.append(ImageFile.from_bytes(
                    file_bytes=part.inline_data.data,
                    file_name=f"image_{len(files)}.png"
                ))

        usage = {
            "model": model_name,
            "prompt_tokens": usage_metadata.prompt_token_count,
            "completion_tokens": usage_metadata.candidates_token_count,
            "total_tokens": usage_metadata.total_token_count
        }

        return Message(
            role="assistant",
            content=text_content.strip(),
            thinking_responses=thoughts,
            files=files,
            function_calls=function_calls,
            usage=usage
        )

    def request_llm(self, model: str, the_conversation: Conversation, functions: List[BaseTool] = None,
                    temperature: float = 0.0, tool_output_callback: Callable = None,
                    additional_parameters: Dict = None, **kwargs) -> Message:
        """
        Sends a request to the Gemini LLM, handling standard chat and function calling.
        """
        functions = functions or []
        additional_parameters = additional_parameters or {}

        converted_tools = [
            types.Tool(function_declarations=[func.to_params(provider="google")])
            for func in functions if isinstance(func, BaseTool)
        ]

        generation_config = self._prepare_generation_config(
            model=model, the_conversation=the_conversation, temperature=temperature,
            tools=converted_tools, additional_parameters=additional_parameters, **kwargs
        )

        while True:
            history = self.convert_conversation_history_to_adapter_format(the_conversation)
            response = self.client.models.generate_content(
                contents=history, model=model, config=generation_config
            )

            candidate = response.candidates[0]
            assistant_message = self._parse_gemini_response(
                response_candidate=candidate, model_name=model, usage_metadata=response.usage_metadata
            )
            the_conversation.messages.append(assistant_message)

            if not assistant_message.function_calls:
                return assistant_message  # Final response from the model

            # --- Handle Function Calling ---
            function_responses = []
            for fc in assistant_message.function_calls:
                function_to_call = next((f for f in functions if f.__name__ == fc.name), None)
                if not function_to_call:
                    raise ValueError(f"Function '{fc.name}' not found in provided tools.")

                try:
                    args = json.loads(fc.arguments)
                    result = function_to_call(**args)
                except Exception as e:
                    result = {"error": f"Execution failed: {e}"}
                    self.logger.error(f"Error executing function '{fc.name}': {e}")

                function_responses.append(FunctionResponse(name=fc.name, response=result, id=fc.id))
                if tool_output_callback:
                    tool_output_callback(fc.name, args, result)

            assistant_message.function_responses = function_responses
            # Loop will continue, sending the function results back to the model

    @property
    def safety_settings(self) -> List[types.SafetySetting]:
        """Returns safety settings to disable all content blocking."""
        return [
            types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for category in (
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            )
        ]

    def generate_image(self, prompt: str, n: int = 1, **kwargs) -> List[ImageFile]:
        """Generates images using the Imagen model."""
        model_name = kwargs.pop('model', self.IMAGEN_DEFAULT_MODEL)
        response = self.client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=n, **kwargs),
        )
        return [
            ImageFile.from_bytes(
                file_bytes=img.image.image_bytes,
                file_name=f"generated_image_{i}.webp"
            )
            for i, img in enumerate(response.generated_images)
        ]

    async def generate_image_async(self, prompt: str, n: int = 1, **kwargs) -> List[ImageFile]:
        """Asynchronously generates images using the Imagen model."""
        response = await self.client.aio.models.generate_images(
            model=self.IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=n, **kwargs)
        )
        return [
            ImageFile.from_bytes(
                file_bytes=img.image.image_bytes,
                file_name=f"generated_image_{i}.webp"
            )
            for i, img in enumerate(response.generated_images)
        ]

    def generate_video(self, prompt: str, aspect_ratio: str = "16:9", person_generation: str = "ALLOW_ADULT",
                       image: ImageFile = None, number_of_videos: int = 1, negative_prompt: str = None,
                       duration_seconds: int = 5) -> List[VideoFile]:
        """Generates videos using the Veo model."""
        params = {"model": self.VEO_MODEL, "prompt": prompt}
        if image:
            params["image"] = types.Image(image_bytes=image.file_bytes, mime_type=f"image/{image.extension}")
            person_generation = "DONT_ALLOW"  # Required for image-to-video
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        config = types.GenerateVideosConfig(
            person_generation=person_generation,
            aspect_ratio=aspect_ratio,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds
        )
        params["config"] = config

        operation = self.client.models.generate_videos(**params)
        self.logger.info(f"Polling for video generation operation: {operation.operation.name}")

        while not operation.done:
            time.sleep(20)  # Polling interval
            operation = self.client.operations.get(operation)

        self.logger.info("Video generation complete.")
        video_files = []
        for i, generated_video in enumerate(operation.response.generated_videos):
            with BytesIO() as buffer:
                generated_video.video.save(buffer)
                buffer.seek(0)
                video_files.append(VideoFile.from_bytes(
                    file_bytes=buffer.read(),
                    file_name=f"generated_video_{i}.mp4"
                ))
        return video_files

    def get_models(self) -> List[str]:
        """Retrieves a list of available models."""
        return [m.name for m in self.client.models.list()]
    
    def request_llm_with_functions(self,
                                   model: str, 
                                   config: genai.types.GenerateContentConfig,
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool]=[], 
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs
                                   ): 
        """
        Not implemented
        This method is not implemented in the GoogleAdapter.
        """
        raise NotImplementedError("request_llm_with_functions is not implemented in GoogleAdapter.")