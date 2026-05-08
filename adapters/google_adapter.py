import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple
from loguru import logger

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


class GoogleAdapter(AdapterBase):
    """
    Adapter for the Gemini API. Built on top of the Interactions API
    (`client.interactions.create`); the legacy `client.models.generate_content`
    surface is no longer used.
    """

    REASONING_EFFORT_MAP = {'high': 24_576, 'medium': 8_000, 'low': 4_000, 'dynamic': -1}
    DEEP_RESEARCH_POLL_INTERVAL_SECONDS = 10
    DEEP_RESEARCH_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}

    # Keys consumed directly when building Interactions params; everything else
    # in additional_parameters is forwarded into generation_config verbatim.
    _RESERVED_PARAM_KEYS = {
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
        "agent_config",
    }

    def __init__(self):
        super().__init__()
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
        from google import genai
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    # ------------------------------------------------------------------
    # File / content conversion
    # ------------------------------------------------------------------

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
            # Gemini direct-PDF processing is capped at ~20 MB / ~3.6k pages
            if file.size < 20_000_000 and file.number_of_pages < 3_600:
                return {
                    "type": "document",
                    "data": file.base64,
                    "mime_type": self._file_mime_type(file),
                }
            logger.info(f"PDF '{file.name}' exceeds size/page limits; sending as text.")
            return {
                "type": "text",
                "text": f'<document name="{file.name}">{file.text}</document>',
            }
        if isinstance(file, (TextDocumentFile, ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
            return {
                "type": "text",
                "text": f'<document name="{file.name}">{file.text}</document>',
            }
        raise TypeError(f"Unsupported file type for Gemini Interactions API: {type(file).__name__}")

    # ------------------------------------------------------------------
    # Conversation → input array
    # ------------------------------------------------------------------

    def _content_items_for_message(self, message: Message) -> List[Dict]:
        items: List[Dict] = []
        if message.content:
            items.append({"type": "text", "text": message.content})
        for file in (message.files or []):
            try:
                items.append(self._convert_file_to_interaction_content(file))
            except TypeError as e:
                logger.warning(e)
        return items

    @staticmethod
    def _normalize_function_arguments(arguments: Any) -> Dict:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments:
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("FunctionCall.arguments is not valid JSON; sending empty dict.")
        return {}

    def convert_conversation_history_to_adapter_format(
        self, the_conversation: Conversation, *args, **kwargs
    ) -> List[Dict]:
        """AdapterBase entry point — delegates to the Interactions-specific
        ``_build_input_from_conversation``."""
        return self._build_input_from_conversation(the_conversation)

    def _build_input_from_conversation(self, conversation: Conversation) -> List[Dict]:
        """
        Builds the ``step_list`` input array consumed by
        ``client.interactions.create``. Every entry is a typed Step:
        ``user_input`` / ``model_output`` for plain exchanges, and
        ``function_call`` / ``function_result`` for prior tool round-trips.
        Role-keyed Turn objects (``{"role": ..., "content": [...]}``) are the
        legacy ``turn_list`` shape and are rejected by the new API.
        """
        input_items: List[Dict] = []
        for message in conversation.messages:
            if message.role == "user":
                content = self._content_items_for_message(message)
                if content:
                    input_items.append({"type": "user_input", "content": content})
                continue

            if message.role == "assistant":
                content = self._content_items_for_message(message)
                if content:
                    input_items.append({"type": "model_output", "content": content})
                for fc in (message.function_calls or []):
                    entry = {
                        "type": "function_call",
                        "name": fc.name,
                        "arguments": self._normalize_function_arguments(fc.arguments),
                    }
                    if fc.id:
                        entry["id"] = fc.id
                    input_items.append(entry)
                for fr in (message.function_responses or []):
                    input_items.append(self._function_result_entry(fr))
                continue

            if message.role == "function":
                for fr in (message.function_responses or []):
                    input_items.append(self._function_result_entry(fr))
                continue

            raise ValueError(f"Invalid message role for Gemini: '{message.role}'")
        return input_items

    @staticmethod
    def _function_result_entry(fr: FunctionResponse) -> Dict:
        entry = {
            "type": "function_result",
            "name": fr.name,
            "result": [{"type": "text", "text": json.dumps(fr.response)}],
        }
        if fr.id:
            entry["call_id"] = fr.id
        return entry

    # ------------------------------------------------------------------
    # Tools / generation_config / response_format
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> List[Dict]:
        tools: List[Dict] = []
        for func in functions or []:
            if not isinstance(func, BaseTool):
                continue
            decl = func.to_params(provider="google")
            tools.append({"type": "function", **decl})

        if additional_parameters.get("web_search"):
            tools.append({"type": "google_search"})
        if additional_parameters.get("url_context"):
            tools.append({"type": "url_context"})
        if additional_parameters.get("code_execution"):
            tools.append({"type": "code_execution"})
        return tools

    def _build_generation_config(
        self,
        model: str,
        additional_parameters: AdditionalParameters,
    ) -> Dict:
        cfg: Dict[str, Any] = {}

        if "temperature" in additional_parameters:
            cfg["temperature"] = additional_parameters["temperature"]
        if "max_output_tokens" in additional_parameters:
            cfg["max_output_tokens"] = additional_parameters["max_output_tokens"]

        if reasoning := additional_parameters.get("reasoning"):
            effort = reasoning.get("effort", "none")
            if "gemini-3" in model:
                cfg["thinking_level"] = effort
            else:
                cfg["thinking_budget"] = self.REASONING_EFFORT_MAP.get(effort, 0)
            cfg["thinking_summaries"] = "auto"

        # Image generation: aspect_ratio + resolution belong inside generation_config.image_config.
        if "image" in (additional_parameters.get("response_modalities") or []):
            aspect_ratio = additional_parameters.get("aspect_ratio")
            resolution = additional_parameters.get("resolution")
            if aspect_ratio and resolution:
                cfg["image_config"] = {
                    "aspect_ratio": aspect_ratio,
                    "image_size": resolution,
                }

        # Forward any unrecognized keys verbatim (matches legacy behavior).
        for key, value in additional_parameters.items():
            if key in self._RESERVED_PARAM_KEYS:
                continue
            cfg[key] = value

        return cfg

    def _build_structured_output(
        self,
        additional_parameters: AdditionalParameters,
    ) -> Dict | None:
        """Returns the polymorphic ``response_format`` body for structured
        output, or None.

        We send this via ``extra_body`` rather than as a named kwarg because
        the installed ``google-genai`` SDK (1.73.x) ships a stale
        ``TextResponseFormatParam`` schema that aliases ``mime_type`` →
        ``mimeType`` on the wire, while the new Interactions server expects
        ``mime_type`` (snake_case) inside the polymorphic dict and
        ``responseFormat`` (camelCase) at the top level. ``extra_body`` keys
        are merged verbatim, so we control the exact wire shape.
        """
        structured_output_class = additional_parameters.get("structured_output")
        if not structured_output_class:
            return None
        if hasattr(structured_output_class, "model_json_schema"):
            raw_schema = structured_output_class.model_json_schema()
            schema = BaseTool.clean_schema(
                BaseTool.resolve_schema_for_google(raw_schema)
            )
        else:
            schema = structured_output_class
        return {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }

    def _build_interaction_kwargs(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> Dict:
        """Assembles the kwargs passed to ``client.interactions.create`` *other
        than* ``input`` and ``previous_interaction_id``. Stays constant across
        the function-calling loop (tools/system_instruction/generation_config
        must be re-supplied on every call per the API contract)."""
        kwargs: Dict[str, Any] = {"model": model}

        if the_conversation.system_prompt:
            kwargs["system_instruction"] = the_conversation.system_prompt

        if tools := self._build_tools(functions, additional_parameters):
            kwargs["tools"] = tools

        if generation_config := self._build_generation_config(model, additional_parameters):
            kwargs["generation_config"] = generation_config

        if response_modalities := additional_parameters.get("response_modalities"):
            kwargs["response_modalities"] = response_modalities

        if response_format := self._build_structured_output(additional_parameters):
            kwargs["extra_body"] = {"response_format": response_format}

        return kwargs

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

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

    def _extract_interaction_steps(
        self,
        steps: List,
    ) -> Tuple[str, List[MediaFile], List[str]]:
        """Walks ``model_output`` steps and pulls text, images, and annotations
        out of their ``content`` arrays (May 2026 steps schema)."""
        text_parts: List[str] = []
        files: List[MediaFile] = []
        additional_responses: List[str] = []
        for step in steps or []:
            if getattr(step, "type", "") != "model_output":
                continue

            for item in getattr(step, "content", []) or []:
                item_type = getattr(item, "type", "")

                if item_type == "image" and getattr(item, "data", None):
                    extension = self._extension_from_mime_type(
                        getattr(item, "mime_type", None),
                        "png",
                    )
                    files.append(ImageFile.from_base64(
                        base64_str=item.data,
                        file_name=f"image_{len(files)}.{extension}",
                    ))
                    continue

                if item_type != "text":
                    continue

                if text := getattr(item, "text", None):
                    text_parts.append(text)

                for annotation in getattr(item, "annotations", []) or []:
                    formatted_annotation = self._format_interaction_annotation(annotation)
                    if formatted_annotation:
                        additional_responses.append(formatted_annotation)

        return "\n\n".join(text_parts).strip(), files, additional_responses

    def _parse_interaction_response(self, interaction, model_name: str) -> Message:
        """Parses a chat ``Interaction`` (text + tools + thinking + code exec)
        into a platform ``Message``."""
        text_parts: List[str] = []
        thoughts: List[ThinkingResponse] = []
        files: List[MediaFile] = []
        function_calls: List[FunctionCall] = []
        additional_responses: List[str] = []

        for step in getattr(interaction, "steps", []) or []:
            step_type = getattr(step, "type", "")

            if step_type == "model_output":
                for item in getattr(step, "content", []) or []:
                    item_type = getattr(item, "type", "")
                    if item_type == "text":
                        if text := getattr(item, "text", None):
                            text_parts.append(text)
                        for ann in getattr(item, "annotations", []) or []:
                            if formatted := self._format_interaction_annotation(ann):
                                additional_responses.append(formatted)
                    elif item_type == "image" and getattr(item, "data", None):
                        extension = self._extension_from_mime_type(
                            getattr(item, "mime_type", None),
                            "png",
                        )
                        files.append(ImageFile.from_base64(
                            base64_str=item.data,
                            file_name=f"image_{len(files)}.{extension}",
                        ))

            elif step_type == "thought":
                summary_pieces = []
                for s in getattr(step, "summary", []) or []:
                    if t := getattr(s, "text", None):
                        summary_pieces.append(t)
                if summary_pieces:
                    thoughts.append(ThinkingResponse(
                        content="".join(summary_pieces),
                        id=getattr(step, "signature", None),
                    ))

            elif step_type == "function_call":
                arguments = getattr(step, "arguments", {}) or {}
                arguments_json = arguments if isinstance(arguments, str) else json.dumps(dict(arguments))
                function_calls.append(FunctionCall(
                    id=getattr(step, "id", None),
                    name=getattr(step, "name", ""),
                    arguments=arguments_json,
                ))

            elif step_type == "code_execution_call":
                code = ""
                args = getattr(step, "arguments", None)
                if args is not None:
                    code = getattr(args, "code", "") or (args.get("code", "") if isinstance(args, dict) else "")
                if code:
                    additional_responses.append(f"# Executable code \n{code}")

            elif step_type == "code_execution_result":
                result = getattr(step, "result", "") or ""
                if result:
                    additional_responses.append(f"# Code execution result \n{result}")

        usage_metadata = getattr(interaction, "usage", None)
        usage = {
            "model": model_name,
            "prompt_tokens": getattr(usage_metadata, "total_input_tokens", None),
            "completion_tokens": getattr(usage_metadata, "total_output_tokens", None),
            "total_tokens": getattr(usage_metadata, "total_tokens", None),
        }

        text_content = "".join(text_parts).strip()
        if not text_content and not function_calls and not files:
            status = getattr(interaction, "status", None)
            error = getattr(interaction, "error", None)
            text_content = f"ERROR. No content in response. status={status} error={error}"

        return Message(
            id=getattr(interaction, "id", None),
            role="assistant",
            content=text_content,
            thinking_responses=thoughts,
            files=files,
            function_calls=function_calls,
            usage=usage,
            additional_responses=additional_responses,
        )

    # ------------------------------------------------------------------
    # Deep Research path
    # ------------------------------------------------------------------

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

        agent_config = dict(additional_parameters.get("agent_config") or {})
        if agent_config:
            agent_config.setdefault("type", "deep-research")
            interaction_params["agent_config"] = agent_config

        return self.client.interactions.create(**interaction_params)

    def _poll_deep_research_interaction(self, interaction):
        while getattr(interaction, "status", None) not in self.DEEP_RESEARCH_TERMINAL_STATUSES:
            time.sleep(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
            interaction = self.client.interactions.get(interaction.id)
        return interaction

    def _parse_deep_research_interaction(self, interaction, model_name: str) -> Message:
        if getattr(interaction, "status", None) != "completed":
            error = getattr(interaction, "error", None)
            raise RuntimeError(
                f"Gemini Deep Research interaction {interaction.id} ended with "
                f"status '{interaction.status}'. {error or ''}".strip()
            )

        text_content, files, additional_responses = self._extract_interaction_steps(
            getattr(interaction, "steps", []) or []
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def request_llm(
            self,
            model: str,
            the_conversation: Conversation,
            functions: List[BaseTool] = None,
            tool_output_callback: Callable = None,
            additional_parameters: AdditionalParameters | None = None,
            **kwargs,
        ) -> Message:
        """
        Sends a request to the Gemini Interactions API, handling chat,
        multimodal input, tool calling, structured output, and image generation.
        Function-calling round-trips inside a single user turn use
        ``previous_interaction_id`` chaining.
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

        base_kwargs = self._build_interaction_kwargs(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            additional_parameters=additional_parameters,
        )

        # Initial call: full conversation history as input.
        input_items = self._build_input_from_conversation(the_conversation)
        interaction = self.client.interactions.create(
            input=input_items,
            **base_kwargs,
        )

        while True:
            assistant_message = self._parse_interaction_response(interaction, model)
            the_conversation.messages.append(assistant_message)

            if not assistant_message.function_calls:
                return assistant_message

            # --- Execute tools and continue with previous_interaction_id ---
            function_responses = []
            for fc in assistant_message.function_calls:
                function_to_call = next((f for f in functions if f.__name__ == fc.name), None)
                if not function_to_call:
                    raise ValueError(f"Function '{fc.name}' not found in provided tools.")

                args: Dict = {}
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

            result_inputs = [self._function_result_entry(fr) for fr in function_responses]
            interaction = self.client.interactions.create(
                input=result_inputs,
                previous_interaction_id=interaction.id,
                **base_kwargs,
            )

    def request_llm_with_functions(self,
                                   model: str,
                                   config: Dict,
                                   the_conversation: Conversation,
                                   functions: List[BaseTool] = [],
                                   tool_output_callback: Callable = None,
                                   additional_parameters: AdditionalParameters | None = None,
                                   **kwargs):
        """Not implemented in GoogleAdapter."""
        raise NotImplementedError("request_llm_with_functions is not implemented in GoogleAdapter.")
