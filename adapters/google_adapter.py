import asyncio
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple
from loguru import logger

from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS
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

    # Background agent polling (e.g. the Antigravity agent). Polling stops both on
    # terminal statuses and on ``requires_action`` (the agent is waiting for a
    # custom-function result before it can continue).
    AGENT_POLL_INTERVAL_SECONDS = 5
    AGENT_POLL_STOP_STATUSES = {
        "completed", "failed", "cancelled", "incomplete", "requires_action",
    }

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
        "agent_config",
    }

    def _build_client(self):
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
        from google import genai
        return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    @property
    def async_client(self):
        """Async surface of the SDK client (``client.aio``): every method is a
        native-async twin of its sync counterpart."""
        return self.client.aio

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

    def _build_input_from_latest_user_message(self, conversation: Conversation) -> List[Dict]:
        """Builds an Interactions ``step_list`` containing only the latest
        ``user`` message. Used for follow-up calls that chain to a prior
        interaction via ``previous_interaction_id`` — the server already holds
        the earlier turns, so re-sending them would duplicate context."""
        latest_user = next(
            (m for m in reversed(conversation.messages) if m.role == "user"),
            None,
        )
        if latest_user is None:
            return []
        content = self._content_items_for_message(latest_user)
        return [{"type": "user_input", "content": content}] if content else []

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
            model_object = self.model_config[model]
            if model_object and model_object["uses_thinking_level"]:
                cfg["thinking_level"] = effort
            else:
                cfg["thinking_budget"] = self.REASONING_EFFORT_MAP.get(effort, 0)
            cfg["thinking_summaries"] = "auto"

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
            # Managed agents surface the finished answer on ``output_text``; fall
            # back to it before declaring an error.
            output_text = (getattr(interaction, "output_text", "") or "").strip()
            if output_text:
                text_content = output_text
            else:
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
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_function_calls(
        self,
        function_calls: List[FunctionCall],
        functions: List[BaseTool],
        tool_output_callback: Callable,
    ) -> List[FunctionResponse]:
        """Executes the model's function calls locally and returns the response records.

        Execution failures are reported back to the model as an error result
        rather than raised, so the conversation can continue.
        """
        function_responses = []
        for fc in function_calls:
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
        return function_responses

    async def _execute_function_calls_async(
        self,
        function_calls: List[FunctionCall],
        functions: List[BaseTool],
        tool_output_callback: Callable,
    ) -> List[FunctionResponse]:
        """Async counterpart of `_execute_function_calls`; additionally awaits coroutine tools."""
        function_responses = []
        for fc in function_calls:
            function_to_call = next((f for f in functions if f.__name__ == fc.name), None)
            if not function_to_call:
                raise ValueError(f"Function '{fc.name}' not found in provided tools.")

            args: Dict = {}
            try:
                args = json.loads(fc.arguments)
                if inspect.iscoroutinefunction(function_to_call):
                    result = await function_to_call(**args)
                else:
                    result = function_to_call(**args)
            except Exception as e:
                result = {"error": f"Execution failed: {e}"}
                logger.error(f"Error executing function '{fc.name}': {e}")

            function_responses.append(FunctionResponse(name=fc.name, response=result, id=fc.id))
            if tool_output_callback:
                tool_output_callback(fc.name, args, result)
        return function_responses

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

    def _deep_research_interaction_params(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters,
    ) -> Dict:
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

        return interaction_params

    def _poll_deep_research_interaction(self, interaction):
        while getattr(interaction, "status", None) not in self.DEEP_RESEARCH_TERMINAL_STATUSES:
            time.sleep(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
            interaction = self.client.interactions.get(interaction.id)
        return interaction

    async def _poll_deep_research_interaction_async(self, interaction):
        """Async counterpart of `_poll_deep_research_interaction`."""
        while getattr(interaction, "status", None) not in self.DEEP_RESEARCH_TERMINAL_STATUSES:
            await asyncio.sleep(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
            interaction = await self.async_client.interactions.get(interaction.id)
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

        interaction = self.client.interactions.create(
            **self._deep_research_interaction_params(model, the_conversation, additional_parameters)
        )
        interaction = self._poll_deep_research_interaction(interaction)
        assistant_message = self._parse_deep_research_interaction(
            interaction=interaction,
            model_name=model,
        )
        the_conversation.messages.append(assistant_message)
        return assistant_message

    async def _request_deep_research_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> Message:
        """Async counterpart of `_request_deep_research`."""
        if functions:
            logger.warning("Gemini Deep Research agents do not support custom function tools.")

        interaction = await self.async_client.interactions.create(
            **self._deep_research_interaction_params(model, the_conversation, additional_parameters)
        )
        interaction = await self._poll_deep_research_interaction_async(interaction)
        assistant_message = self._parse_deep_research_interaction(
            interaction=interaction,
            model_name=model,
        )
        the_conversation.messages.append(assistant_message)
        return assistant_message

    # ------------------------------------------------------------------
    # Antigravity agent path
    # ------------------------------------------------------------------

    def _build_antigravity_kwargs(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> Dict:
        """Kwargs for ``client.interactions.create`` on the Antigravity agent
        path, excluding ``input`` / ``previous_interaction_id`` / ``environment``.

        Unlike the standard Gemini chat path, the Antigravity agent rejects
        ``generation_config`` parameters (``temperature``, ``max_output_tokens``,
        ...) and structured output with a 400, so neither is sent. Built-in tools
        (``google_search`` / ``url_context`` / ``code_execution``) and any custom
        functions are forwarded; the sandbox filesystem is enabled implicitly by
        the ``environment`` argument supplied at call time."""
        kwargs: Dict[str, Any] = {"agent": model}
        if the_conversation.system_prompt:
            kwargs["system_instruction"] = the_conversation.system_prompt
        if tools := self._build_tools(functions, additional_parameters):
            kwargs["tools"] = tools
        return kwargs

    def _poll_agent_interaction(self, interaction):
        """Polls a background interaction until it reaches a terminal status or
        ``requires_action`` (waiting on a client-side function result)."""
        while getattr(interaction, "status", None) not in self.AGENT_POLL_STOP_STATUSES:
            time.sleep(self.AGENT_POLL_INTERVAL_SECONDS)
            interaction = self.client.interactions.get(interaction.id)
        return interaction

    async def _poll_agent_interaction_async(self, interaction):
        """Async counterpart of `_poll_agent_interaction`."""
        while getattr(interaction, "status", None) not in self.AGENT_POLL_STOP_STATUSES:
            await asyncio.sleep(self.AGENT_POLL_INTERVAL_SECONDS)
            interaction = await self.async_client.interactions.get(interaction.id)
        return interaction

    def _request_antigravity(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
        additional_parameters: AdditionalParameters,
    ) -> Message:
        """Runs the Antigravity managed agent through the Gemini Interactions API.

        A single ``interactions.create`` provisions a remote Linux sandbox
        (``environment="remote"``) and runs the agent's internal tool-use loop
        (code execution, web search, URL fetch, filesystem) server-side, returning
        the finished result. Only *custom* functions need a client-side
        round-trip: those are executed locally and fed back via
        ``previous_interaction_id`` (function calling is stateful-only), reusing
        the same sandbox ``environment``. Built-in and filesystem calls are run by
        the sandbox and already carry their results, so they are not re-executed.

        When the model is flagged ``background_mode: true`` the interaction runs
        asynchronously (``background=True`` + ``store=True``) and is polled until
        it completes or needs a function result — the recommended mode for these
        long-running agent tasks.
        """
        base_kwargs = self._build_antigravity_kwargs(
            model, the_conversation, functions, additional_parameters
        )
        model_object = self.model_config[model]
        background = bool(model_object and model_object["background_mode"])
        if background:
            base_kwargs["background"] = True
            base_kwargs["store"] = True

        custom_function_names = {f.__name__ for f in functions}

        interaction = self.client.interactions.create(
            input=self._build_input_from_conversation(the_conversation),
            environment="remote",
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            if background:
                interaction = self._poll_agent_interaction(interaction)

            assistant_message = self._parse_interaction_response(interaction, model)

            # Keep only the calls the platform is responsible for executing.
            custom_calls = [
                fc for fc in assistant_message.function_calls
                if fc.name in custom_function_names
            ]
            assistant_message.function_calls = custom_calls
            the_conversation.messages.append(assistant_message)

            if not custom_calls:
                return assistant_message

            function_responses = self._execute_function_calls(
                custom_calls, functions, tool_output_callback
            )
            assistant_message.function_responses = function_responses

            interaction = self.client.interactions.create(
                input=[self._function_result_entry(fr) for fr in function_responses],
                previous_interaction_id=interaction.id,
                environment=getattr(interaction, "environment_id", None) or "remote",
                **base_kwargs,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

    async def _request_antigravity_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
        additional_parameters: AdditionalParameters,
    ) -> Message:
        """Async counterpart of `_request_antigravity` (same flow on `client.aio`)."""
        base_kwargs = self._build_antigravity_kwargs(
            model, the_conversation, functions, additional_parameters
        )
        model_object = self.model_config[model]
        background = bool(model_object and model_object["background_mode"])
        if background:
            base_kwargs["background"] = True
            base_kwargs["store"] = True

        custom_function_names = {f.__name__ for f in functions}

        interaction = await self.async_client.interactions.create(
            input=self._build_input_from_conversation(the_conversation),
            environment="remote",
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            if background:
                interaction = await self._poll_agent_interaction_async(interaction)

            assistant_message = self._parse_interaction_response(interaction, model)

            # Keep only the calls the platform is responsible for executing.
            custom_calls = [
                fc for fc in assistant_message.function_calls
                if fc.name in custom_function_names
            ]
            assistant_message.function_calls = custom_calls
            the_conversation.messages.append(assistant_message)

            if not custom_calls:
                return assistant_message

            function_responses = await self._execute_function_calls_async(
                custom_calls, functions, tool_output_callback
            )
            assistant_message.function_responses = function_responses

            interaction = await self.async_client.interactions.create(
                input=[self._function_result_entry(fr) for fr in function_responses],
                previous_interaction_id=interaction.id,
                environment=getattr(interaction, "environment_id", None) or "remote",
                **base_kwargs,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

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
        multimodal input, tool calling, and structured output.

        Server-side conversation state is reused across turns via
        ``previous_interaction_id``: on the first turn the full conversation is
        sent and the returned ``interaction.id`` is stored on the assistant
        ``Message``; on every subsequent turn (and on every function-calling
        round-trip inside a turn) only the new ``user_input`` / new
        ``function_result`` entries are sent, chained to the prior interaction
        id so the server retrieves the rest of the history.
        """
        functions = functions or []
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

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

        if model_object and model_object["agent_type"] == "antigravity":
            return self._request_antigravity(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )

        base_kwargs = self._build_interaction_kwargs(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            additional_parameters=additional_parameters,
        )

        # Server-side conversation state: if a prior assistant message in this
        # conversation has an interaction id, chain to it via
        # ``previous_interaction_id`` and send only the new user_input.
        # Otherwise (first turn), send the full history.
        prev_interaction_id = the_conversation.last_assistant_id
        if prev_interaction_id:
            input_items = self._build_input_from_latest_user_message(the_conversation)
            interaction = self.client.interactions.create(
                input=input_items,
                previous_interaction_id=prev_interaction_id,
                **base_kwargs,
            )
        else:
            input_items = self._build_input_from_conversation(the_conversation)
            interaction = self.client.interactions.create(
                input=input_items,
                **base_kwargs,
            )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            assistant_message = self._parse_interaction_response(interaction, model)
            the_conversation.messages.append(assistant_message)

            if not assistant_message.function_calls:
                return assistant_message

            # --- Execute tools and continue with previous_interaction_id ---
            function_responses = self._execute_function_calls(
                assistant_message.function_calls, functions, tool_output_callback
            )
            assistant_message.function_responses = function_responses

            result_inputs = [self._function_result_entry(fr) for fr in function_responses]
            interaction = self.client.interactions.create(
                input=result_inputs,
                previous_interaction_id=interaction.id,
                **base_kwargs,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

    async def request_llm_async(
            self,
            model: str,
            the_conversation: Conversation,
            functions: List[BaseTool] = None,
            tool_output_callback: Callable = None,
            additional_parameters: AdditionalParameters | None = None,
            **kwargs,
        ) -> Message:
        """
        Async counterpart of `request_llm`, backed by the SDK's native async
        surface (``client.aio``). Follows the same dispatch rules: Deep
        Research and Antigravity agent models route to their dedicated paths,
        everything else runs the standard Interactions chat/tool loop with
        ``previous_interaction_id`` chaining.
        """
        functions = functions or []
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        model_object = self.model_config[model]
        if (
            model_object
            and model_object["background_mode"]
            and model_object["agent_type"] == "deep_research"
        ):
            return await self._request_deep_research_async(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                additional_parameters=additional_parameters,
            )

        if model_object and model_object["agent_type"] == "antigravity":
            return await self._request_antigravity_async(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )

        base_kwargs = self._build_interaction_kwargs(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            additional_parameters=additional_parameters,
        )

        prev_interaction_id = the_conversation.last_assistant_id
        if prev_interaction_id:
            input_items = self._build_input_from_latest_user_message(the_conversation)
            interaction = await self.async_client.interactions.create(
                input=input_items,
                previous_interaction_id=prev_interaction_id,
                **base_kwargs,
            )
        else:
            input_items = self._build_input_from_conversation(the_conversation)
            interaction = await self.async_client.interactions.create(
                input=input_items,
                **base_kwargs,
            )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            assistant_message = self._parse_interaction_response(interaction, model)
            the_conversation.messages.append(assistant_message)

            if not assistant_message.function_calls:
                return assistant_message

            # --- Execute tools and continue with previous_interaction_id ---
            function_responses = await self._execute_function_calls_async(
                assistant_message.function_calls, functions, tool_output_callback
            )
            assistant_message.function_responses = function_responses

            result_inputs = [self._function_result_entry(fr) for fr in function_responses]
            interaction = await self.async_client.interactions.create(
                input=result_inputs,
                previous_interaction_id=interaction.id,
                **base_kwargs,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

