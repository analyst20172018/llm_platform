import json
import os
import time
from io import BytesIO
from typing import Callable, Dict, List, Tuple
from enum import Enum
from loguru import logger

from google.genai import types

from .adapter_base import AdapterBase
from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile)
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters

class ImagenModel(Enum):
    STANDARD = "imagen-4.0-generate-001"
    ULTRA = "imagen-4.0-ultra-generate-001"

class ImagenImageSize(Enum):
    _1K = "1K"
    _2K = "2K"

class GoogleAdapter(AdapterBase):
    """
    Adapter for interacting with the Google Gemini API.
    """
    # Class-level constants for configuration and mapping
    GEMINI_ROLE_MAPPING = {'user': 'user', 'assistant': 'model'}
    REASONING_EFFORT_MAP = {'high': 24_576, 'medium': 8_000, 'low': 4_000, 'dynamic': -1}
    VEO_MODEL = 'veo-3.1-generate-preview'
    DEEP_RESEARCH_POLL_INTERVAL_SECONDS = 10

    def __init__(self):
        super().__init__()
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
        from google import genai
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    def _convert_file_to_part(self, file: MediaFile, id=None) -> types.Part:
        """Converts a MediaFile object to a Gemini API Part."""
        part = None
        if isinstance(file, ImageFile):
            part = types.Part.from_bytes(
                    data=file.file_bytes, 
                    mime_type=f"image/{file.extension}"
            )
        elif isinstance(file, AudioFile):
            part = types.Part.from_bytes(data=file.file_bytes, mime_type="audio/mp3")
        elif isinstance(file, (TextDocumentFile, ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
            text = f'<document name="{file.name}">{file.text}</document>'
            part = types.Part.from_text(text=text)
        elif isinstance(file, PDFDocumentFile):
            # Gemini has limits on direct PDF processing
            if file.size < 20_000_000 and file.number_of_pages < 3_600:
                part = types.Part.from_bytes(data=file.bytes, mime_type="application/pdf")
            else:
                logger.info(f"PDF '{file.name}' exceeds size/page limits; sending as text.")
                text = f'<document name="{file.name}">{file.text}</document>'
                part = types.Part.from_text(text=text)
        else: 
            logger.critical(f"Unsupported file type for Gemini: {type(file).__name__}")
            raise TypeError(f"Unsupported file type for Gemini: {type(file).__name__}")
        
        if id:
            part.thought_signature = id

        return part

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
                part = types.Part.from_text(text=message.content)
                if message.id:
                    part.thought_signature = message.id
                parts.append(part)

            if message.files:
                for file in message.files:
                    try:
                        parts.append(self._convert_file_to_part(file, id=message.id))
                    except TypeError as e:
                        logger.warning(e)

            if message.function_calls:
                for fc in message.function_calls:
                    part = types.Part.from_function_call(name=fc.name, args=json.loads(fc.arguments))
                    if fc.id:
                        part.thought_signature = fc.id
                    parts.append(part)

            if parts:
                history.append(types.Content(role=role, parts=parts))

            # If an assistant message has function responses, add a subsequent 'function' role message
            if message.function_responses:
                response_parts = [
                    types.Part.from_function_response(name=fr.name, response=fr.response)
                    for fr in message.function_responses
                ]
                if response_parts:
                    history.append(types.Content(role="user", parts=response_parts))
        return history

    @staticmethod
    def _file_mime_type(file: BaseFile) -> str:
        extension = file.extension
        if isinstance(file, ImageFile):
            return f"image/{extension}"
        if isinstance(file, AudioFile):
            return "audio/mp3"
        if isinstance(file, VideoFile):
            if extension == "3gp":
                return "video/3gpp"
            if extension == "mpg":
                return "video/mpeg"
            return f"video/{extension}"
        if isinstance(file, PDFDocumentFile):
            return "application/pdf"
        return "text/plain"

    def _convert_file_to_interaction_content(self, file: BaseFile) -> Dict:
        if isinstance(file, ImageFile):
            return {
                "type": "image",
                "data": file.base64,
                "mime_type": self._file_mime_type(file),
            }
        if isinstance(file, AudioFile):
            return {
                "type": "audio",
                "data": file.base64,
                "mime_type": self._file_mime_type(file),
            }
        if isinstance(file, VideoFile):
            return {
                "type": "video",
                "data": file.base64,
                "mime_type": self._file_mime_type(file),
            }
        if isinstance(file, PDFDocumentFile):
            return {
                "type": "document",
                "data": file.base64,
                "mime_type": self._file_mime_type(file),
            }
        if isinstance(file, (TextDocumentFile, ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
            return {
                "type": "text",
                "text": f'<document name="{file.name}">{file.text}</document>',
            }

        raise TypeError(f"Unsupported file type for Gemini Deep Research: {type(file).__name__}")

    def _convert_conversation_to_interaction_input(self, conversation: Conversation) -> str | List[Dict]:
        user_messages = [message for message in conversation.messages if message.role == "user"]
        if not user_messages:
            return ""

        latest_message = user_messages[-1]
        text_input = latest_message.content
        if conversation.system_prompt:
            text_input = (
                f"System instructions:\n{conversation.system_prompt}\n\n"
                f"User request:\n{text_input}"
            )

        if not latest_message.files:
            return text_input

        interaction_input: List[Dict] = []
        if text_input:
            interaction_input.append({"type": "text", "text": text_input})

        for file in latest_message.files:
            interaction_input.append(self._convert_file_to_interaction_content(file))

        return interaction_input

    def _prepare_interaction_tools(self, additional_parameters: AdditionalParameters) -> List[Dict]:
        return []

    def _create_deep_research_interaction(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters,
    ):
        interaction_params = {
            "input": self._convert_conversation_to_interaction_input(the_conversation),
            "agent": model,
            "background": True,
            "store": True,
        }

        if tools := self._prepare_interaction_tools(additional_parameters):
            interaction_params["tools"] = tools

        agent_config = dict(additional_parameters.get("agent_config") or {})
        if agent_config:
            agent_config.setdefault("type", "deep-research")
            interaction_params["agent_config"] = agent_config

        return self.client.interactions.create(**interaction_params)

    def _poll_deep_research_interaction(self, interaction):
        while getattr(interaction, "status", None) in {"queued", "in_progress"}:
            time.sleep(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
            interaction = self.client.interactions.get(interaction.id)

        return interaction

    @staticmethod
    def _format_interaction_annotation(annotation) -> str | None:
        annotation_type = getattr(annotation, "type", "")
        if annotation_type == "url_citation":
            url = getattr(annotation, "url", None)
            if not url:
                return None
            title = getattr(annotation, "title", None) or url
            return f"Citation: {title} - {url}"
        if annotation_type == "file_citation":
            file_name = getattr(annotation, "file_name", None)
            source = getattr(annotation, "source", None) or getattr(annotation, "document_uri", None)
            if not file_name and not source:
                return None
            if file_name and source:
                return f"File citation: {file_name} - {source}"
            return f"File citation: {file_name or source}"
        if annotation_type == "place_citation":
            name = getattr(annotation, "name", None)
            url = getattr(annotation, "url", None)
            if name and url:
                return f"Place citation: {name} - {url}"
            return f"Place citation: {name or url}" if name or url else None
        return None

    @staticmethod
    def _extension_from_mime_type(mime_type: str | None, default: str) -> str:
        if not mime_type or "/" not in mime_type:
            return default
        extension = mime_type.split("/", 1)[1]
        return "jpg" if extension == "jpeg" else extension

    def _extract_interaction_outputs(
        self,
        outputs: List,
    ) -> Tuple[str, List[MediaFile], List[str]]:
        text_parts = []
        files = []
        additional_responses = []
        for output in outputs or []:
            output_type = getattr(output, "type", "")

            if output_type == "image" and getattr(output, "data", None):
                extension = self._extension_from_mime_type(
                    getattr(output, "mime_type", None),
                    "png",
                )
                files.append(ImageFile.from_base64(
                    base64_str=output.data,
                    file_name=f"image_{len(files)}.{extension}",
                ))
                continue

            if output_type != "text":
                continue

            if text := getattr(output, "text", None):
                text_parts.append(text)

            for annotation in getattr(output, "annotations", []) or []:
                formatted_annotation = self._format_interaction_annotation(annotation)
                if formatted_annotation:
                    additional_responses.append(formatted_annotation)

        return "\n\n".join(text_parts).strip(), files, additional_responses

    def _parse_deep_research_interaction(self, interaction, model_name: str) -> Message:
        if getattr(interaction, "status", None) != "completed":
            error = getattr(interaction, "error", None)
            raise RuntimeError(
                f"Gemini Deep Research interaction {interaction.id} ended with "
                f"status '{interaction.status}'. {error or ''}".strip()
            )

        text_content, files, additional_responses = self._extract_interaction_outputs(
            getattr(interaction, "outputs", []) or []
        )

        usage_metadata = getattr(interaction, "usage", None)
        usage = {
            "model": model_name,
            "prompt_tokens": getattr(usage_metadata, "total_input_tokens", None),
            "completion_tokens": getattr(usage_metadata, "total_output_tokens", None),
            "total_tokens": getattr(usage_metadata, "total_tokens", None),
        }

        return Message(
            id=interaction.id,
            role="assistant",
            content=text_content,
            usage=usage,
            files=files,
            additional_responses=additional_responses,
        )

    def _request_deep_research(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> Message:
        if functions:
            logger.warning("Gemini Deep Research agents do not support custom function tools.")

        interaction = self._create_deep_research_interaction(
            model=model,
            the_conversation=the_conversation,
            additional_parameters=additional_parameters,
        )
        interaction = self._poll_deep_research_interaction(interaction)
        assistant_message = self._parse_deep_research_interaction(
            interaction=interaction,
            model_name=model,
        )
        the_conversation.messages.append(assistant_message)
        return assistant_message

    def _prepare_generation_config(
        self,
        model: str,
        the_conversation: Conversation,
        tools: List,
        additional_parameters: AdditionalParameters,
    ) -> types.GenerateContentConfig:
        """Prepares the GenerateContentConfig for a Gemini API call."""
        config_params = {
            "tools": tools,
            "safety_settings": self.safety_settings,
        }

        if "max_output_tokens" in additional_parameters:
            config_params["max_output_tokens"] = additional_parameters["max_output_tokens"]

        if "temperature" in additional_parameters:
            config_params["temperature"] = additional_parameters["temperature"]

        if the_conversation.system_prompt:
            config_params["system_instruction"] = the_conversation.system_prompt

        # Reasoning effort / Thinking config
        if reasoning_effort_parameter := additional_parameters.get("reasoning", {}):
            reasoning_effort = reasoning_effort_parameter.get('effort', 'none')
            if 'gemini-3' in model:
                # Gemini 3 introduces new parameter - Thinking level
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort,
                    include_thoughts=True
                )
            else:
                # If reasoning effort is not set, then disable thinking, by setting the parameter to 0
                thinking_budget = self.REASONING_EFFORT_MAP.get(reasoning_effort, 0)
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget, 
                    include_thoughts=True
                )

        if "response_modalities" in additional_parameters:
            config_params["response_modalities"] = additional_parameters["response_modalities"]
            if "image" in config_params["response_modalities"]:
                if (aspect_ratio := additional_parameters.get("aspect_ratio")) and \
                   (resolution := additional_parameters.get("resolution")):
                    config_params["image_config"] = types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    )

        if additional_parameters.get("web_search"):
            config_params["tools"].append(types.Tool(google_search=types.GoogleSearch()))
        if additional_parameters.get("url_context"):
            config_params["tools"].append(types.Tool(url_context=types.UrlContext()))
        if additional_parameters.get("code_execution"):
            config_params["tools"].append(types.Tool(code_execution=types.ToolCodeExecution))

        # Structured output
        if structured_output_class := additional_parameters.get("structured_output", None):
            config_params["response_mime_type"] = "application/json"
            # If it's a Pydantic model, convert to a Gemini-compatible dict schema
            if hasattr(structured_output_class, "model_json_schema"):
                raw_schema = structured_output_class.model_json_schema()
                config_params["response_schema"] = BaseTool.clean_schema(
                    BaseTool.resolve_schema_for_google(raw_schema)
                )
            else:
                config_params["response_schema"] = structured_output_class

        reserved_keys = {
            "max_output_tokens",
            "temperature",
            "reasoning",
            "response_modalities",
            "web_search",
            "url_context",
            "code_execution",
            "structured_output",
            "aspect_ratio",
            "resolution",
        }
        for key, value in additional_parameters.items():
            if key in reserved_keys:
                continue
            config_params[key] = value

        return types.GenerateContentConfig(**config_params)

    def _parse_gemini_response(self, response: types.HttpResponse, model_name: str) -> Message:
        """Parses a Gemini API response candidate into a Message object."""
        text_content, thoughts, files, function_calls, additional_responses = "", [], [], [], []

        usage_metadata=response.usage_metadata

        if not response.candidates or len(response.candidates) == 0:
            # Get the block reason from prompt_feedback=GenerateContentResponsePromptFeedback(block_reason=...)
            if prompt_feedback := getattr(response, 'prompt_feedback', None):
                block_reason = getattr(prompt_feedback, 'block_reason', '')
            else:
                block_reason = ''
            text_content = f"ERROR. No candidates in response. {block_reason}"
            thought_signature = None

        else:
            response_candidate = response.candidates[0]

            if response_candidate.content.parts:
                for part in response_candidate.content.parts:
                    
                    # There is a new thought_signature field in Part since Gemini 3
                    thought_signature = getattr(part, "thought_signature", None) or None

                    if fc := part.function_call:
                        function_calls.append(FunctionCall(
                            id=thought_signature,
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args))
                        ))
                    elif part.text:
                        if part.thought:
                            thoughts.append(ThinkingResponse(content=part.text, id=None))
                        else:
                            text_content += part.text
                    elif part.inline_data and "image" in part.inline_data.mime_type:
                        
                        mime_type = part.inline_data.mime_type  # e.g. "image/png"
                        file_type = mime_type.split("/")[-1]
                        
                        files.append(ImageFile.from_bytes(
                            file_bytes=part.inline_data.data,
                            file_name=f"image_{len(files)}.{file_type}"
                        ))
                    elif part.executable_code:
                        additional_responses.append("# Executable code \n" + part.executable_code.code)
                    elif part.code_execution_result:
                        additional_responses.append("# Code execution result \n" + part.code_execution_result.output)
            # There are no parts in the candidate in the response
            else:
                logger.warning(f"No parts in the response candidate. Finish reason is {response_candidate.finish_reason}")
                text_content = f"No content in response. Finish reason is {response_candidate.finish_reason}"
                thought_signature = response.response_id

        usage = {
            "model": model_name,
            "prompt_tokens": usage_metadata.prompt_token_count,
            "completion_tokens": usage_metadata.candidates_token_count,
            "total_tokens": usage_metadata.total_token_count
        }

        return Message(
            id=thought_signature,
            role="assistant",
            content=text_content.strip(),
            thinking_responses=thoughts,
            files=files,
            function_calls=function_calls,
            usage=usage,
            additional_responses=additional_responses,
        )

    def request_llm(
            self, 
            model: str, 
            the_conversation: Conversation, 
            functions: List[BaseTool] = None,
            tool_output_callback: Callable = None,
            additional_parameters: AdditionalParameters | None = None, 
            **kwargs
        ) -> Message:
        """
        Sends a request to the Gemini LLM, handling standard chat and function calling.
        """
        functions = functions or []
        additional_parameters = additional_parameters or {}

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)

        model_object = self.model_config[model]
        if (
            model_object
            and model_object["background_mode"]
            and model_object["agent_type"] == "deep_research"
        ):
            return self._request_deep_research(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                additional_parameters=additional_parameters,
            )

        function_declarations = [
            func.to_params(provider="google")
            for func in functions if isinstance(func, BaseTool)
        ]
        converted_tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else []

        generation_config = self._prepare_generation_config(
            model=model,
            the_conversation=the_conversation,
            tools=converted_tools,
            additional_parameters=additional_parameters,
        )

        while True:
            history = self.convert_conversation_history_to_adapter_format(the_conversation)

            response = self.client.models.generate_content(
                contents=history, model=model, config=generation_config
            )

            assistant_message = self._parse_gemini_response(
                response=response, model_name=model
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
                    logger.error(f"Error executing function '{fc.name}': {e}")

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

        # Define the model and image size based on quality
        quality = kwargs.pop('quality', 'high')
        if quality == 'high':
            model_name = ImagenModel.ULTRA.value
            image_size = ImagenImageSize._2K.value
        elif quality == 'medium':
            model_name = ImagenModel.STANDARD.value
            image_size = ImagenImageSize._2K.value
        elif quality == 'low':
            model_name = ImagenModel.STANDARD.value
            image_size = ImagenImageSize._1K.value
        else:
            model_name = ImagenModel.ULTRA.value
            image_size = ImagenImageSize._2K.value

        response = self.client.models.generate_images(
            model=model_name,
            prompt=prompt,
            #config=types.GenerateImagesConfig(
            config=dict(
                number_of_images=n, 
                image_size=image_size,
                **kwargs
            ),
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

    def generate_video(self, 
                       prompt: str, 
                       model: str = VEO_MODEL,
                       aspect_ratio: str = "16:9",                  # "16:9" (default, 720p & 1080p), "9:16"(720p & 1080p)
                       person_generation: str = "allow_all",        # Text-to-video & Extension: "allow_all" only; Image-to-video, Interpolation, & Reference images: "allow_adult" only
                       resolution: str = "1080p",                   # "720p" (default), "1080p" (only supports 8s duration and 16:9) 
                       duration_seconds: int = 8,                   # "4", "6", "8". Must be "8" when using extension or interpolation (supports both 16:9 and 9:16), and when using referenceImages (only supports 16:9)
                       negative_prompt: str = None,
                       image: ImageFile = None,                     # An initial image to animate.
                       number_of_videos: int = 1,                   # Number of videos to generate. Default is 1, max is 4.
                       #generate_audio: bool = True,                 # Whether to generate audio with the video. Default is True. NOT SUPPORTED
                       seed: int = None,                            # Random seed for generation. Default is None (random).
                       lastFrame: ImageFile = None,                 # The final image for an interpolation video to transition. Must be used in combination with the image parameter.
                       video: VideoFile = None,                     # Video to be used for video extension (only support resolution: 720p, up to 20 times)
                       reference_images: List[ImageFile] = None,    # Up to three images to be used as style and content references.
                    ) -> List[VideoFile]:
        """Generates videos using the Veo model."""
        params = {
            "model": model, 
            "prompt": prompt,
        }
        if image:
            params["image"] = types.Image(image_bytes=image.file_bytes, mime_type=f"image/{image.extension}")
            if person_generation != "allow_adult":
                logger.warning("When using 'image' parameter, 'person_generation' must be set to 'allow_adult'. Overriding.")
                person_generation = "allow_adult"

        if video:
            params["source"] = types.Video(video_bytes=video.file_bytes, mime_type=f"video/{video.extension}") # Use `source` instead of `video`
            if duration_seconds != 8:
                logger.warning("When using 'video' parameter, 'duration_seconds' must be set to 8. Overriding.")
                duration_seconds = 8
            if resolution != "720p":
                logger.warning("When using 'video' parameter, 'resolution' must be set to '720p'. Overriding.")
                resolution = "720p"

        config = types.GenerateVideosConfig(
            number_of_videos=number_of_videos,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
            person_generation=person_generation,
            #generate_audio=generate_audio,
        )
        if negative_prompt:
            config.negative_prompt = negative_prompt
        if seed:
            config.seed = seed
        if reference_images:
            if len(reference_images) > 3:
                raise ValueError("A maximum of 3 reference images can be provided.")
            config.reference_images = [
                types.Image(image_bytes=img.file_bytes, mime_type=f"image/{img.extension}")
                for img in reference_images
            ]
        if lastFrame:
            if not image:
                raise ValueError("The 'lastFrame' parameter requires the 'image' parameter to be set.")
            config.last_frame = types.Image(image_bytes=lastFrame.file_bytes, mime_type=f"image/{lastFrame.extension}")
        params["config"] = config

        operation = self.client.models.generate_videos(**params)

        while not operation.done:
            time.sleep(20)  # Polling interval
            operation = self.client.operations.get(operation)

        video_files = []
        for i, generated_video in enumerate(operation.response.generated_videos):
            self.client.files.download(file=generated_video.video)
            buffer = BytesIO(generated_video.video.video_bytes)
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
                                   config: types.GenerateContentConfig,
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool]=[], 
                                   tool_output_callback: Callable=None,
                                   additional_parameters: AdditionalParameters | None = None,
                                   **kwargs
                                   ): 
        """
        Not implemented
        This method is not implemented in the GoogleAdapter.
        """
        raise NotImplementedError("request_llm_with_functions is not implemented in GoogleAdapter.")
