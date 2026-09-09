import asyncio
import base64
import copy
import io
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple
from loguru import logger
from pydantic import TypeAdapter

from .response_metadata import google_metadata, result_status
from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS
from .serializers import provider_dump, missing_continuation
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
    PDF_MAX_BYTES = 50_000_000
    PDF_MAX_PAGES = 1_000
    # Leave room for SDK-added fields within the 20 MB inline HTTP limit.
    INLINE_REQUEST_MAX_BYTES = 19_000_000

    # Background agent polling (e.g. the Antigravity agent). Polling stops both on
    # terminal statuses and on ``requires_action`` (the agent is waiting for a
    # custom-function result before it can continue).
    AGENT_POLL_INTERVAL_SECONDS = 5

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
        return file.mime_type

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
            if file.size > self.PDF_MAX_BYTES or file.number_of_pages > self.PDF_MAX_PAGES:
                raise ValueError(f"PDF '{file.name}' exceeds Gemini's 50 MB / 1,000 page limit.")
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
        raise TypeError(f"Unsupported file type for Gemini Interactions API: {type(file).__name__}")

    # ------------------------------------------------------------------
    # Conversation → input array
    # ------------------------------------------------------------------

    def _content_items_for_message(self, message: Message) -> List[Dict]:
        items: List[Dict] = []
        if message.content:
            items.append({"type": "text", "text": message.content})
        for file in (message.files or []):
            items.append(self._convert_file_to_interaction_content(file))
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

    def _build_input_from_conversation(self, conversation: Conversation, model: str | None = None) -> List[Dict]:
        """
        Builds the ``step_list`` input array consumed by
        ``client.interactions.create``. Every entry is a typed Step:
        ``user_input`` / ``model_output`` for plain exchanges, and
        ``function_call`` / ``function_result`` for prior tool round-trips.
        Role-keyed Turn objects (``{"role": ..., "content": [...]}``) are the
        legacy ``turn_list`` shape and are rejected by the new API.
        """
        input_items: List[Dict] = []
        pdf_pages = sum(
            file.number_of_pages
            for message in conversation.messages for file in message.files or []
            if isinstance(file, PDFDocumentFile)
        )
        if pdf_pages > self.PDF_MAX_PAGES:
            raise ValueError("Gemini accepts at most 1,000 PDF pages in one request.")
        for message in conversation.messages:
            if message.role == "user":
                content = self._content_items_for_message(message)
                if content:
                    input_items.append({"type": "user_input", "content": content})
                input_items.extend(self._function_result_entry(fr) for fr in message.function_responses)
                continue

            if message.role == "assistant":
                # Interactions explicitly supports thought replay across Gemini models.
                native = message.replay_data("google")
                if "steps" in native:
                    input_items.extend(native["steps"])
                    input_items.extend(self._function_result_entry(fr) for fr in message.function_responses)
                    continue
                content = self._content_items_for_message(message)
                if content:
                    input_items.append({"type": "model_output", "content": content})
                for fc in (message.function_calls or []):
                    entry = {
                        "type": "function_call",
                        "name": fc.name,
                        "arguments": self._normalize_function_arguments(fc.arguments),
                    }
                    if fc.call_id:
                        entry["id"] = fc.call_id
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

    def _function_result_entry(self, fr: FunctionResponse) -> Dict:
        result = [{"type": "text", "text": json.dumps(fr.response)}]
        for file in fr.files:
            if isinstance(file, ImageFile):
                result.append(self._convert_file_to_interaction_content(file))
            elif isinstance(file, DocumentFile):
                result.append({"type": "text", "text": f"{file.name}\n{file.text}"})
            else:
                raise ValueError(f"Unsupported Gemini tool attachment: {type(file).__name__}")
        entry = {"type": "function_result", "name": fr.name, "result": result}
        if fr.call_id:
            entry["call_id"] = fr.call_id
        return entry

    def _continuation_input(self, conversation, model, *, antigravity=False):
        state = conversation.continuation("google", model)
        messages = conversation.messages[state["length"]:] if state else conversation.messages
        params = {"input": self._build_input_from_conversation(Conversation(messages), model)}
        if state:
            params["previous_interaction_id"] = state["response_id"]
        if antigravity:
            if state and not state.get("environment_id"):
                raise ValueError("Antigravity continuation is missing its environment ID; reset continuation to start a new environment.")
            params["environment"] = state["environment_id"] if state else "remote"
        return params

    def _replay_interaction_kwargs(self, conversation, model, kwargs):
        params = {key: value for key, value in kwargs.items() if key != "previous_interaction_id"}
        params["input"] = self._build_input_from_conversation(conversation, model)
        model_object = self.model_config[model]
        if model_object and model_object["agent_type"] == "deep_research":
            self._prepend_research_instructions(params["input"], conversation)
        return params

    def _pdf_upload_candidates(self, kwargs):
        """Choose largest inline PDFs until the encoded request fits the budget.

        Work on detached wire dictionaries. Uploaded URIs live only in this
        request; a later full-history replay reuploads the original bytes.
        """
        documents = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("type") == "document" and value.get("mime_type") == "application/pdf" and value.get("data"):
                    documents.append(value)
                else:
                    for child in value.values():
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(kwargs.get("input"))
        request_size = len(json.dumps(kwargs, ensure_ascii=True).encode("utf-8"))
        candidates = []
        for document in sorted(documents, key=lambda doc: len(doc["data"]), reverse=True):
            if request_size <= self.INLINE_REQUEST_MAX_BYTES:
                break
            candidates.append(document)
            request_size -= len(document["data"]) - 1024  # conservative URI overhead
        if candidates and request_size > self.INLINE_REQUEST_MAX_BYTES:
            raise ValueError("Gemini request exceeds the inline size limit even after PDF uploads.")
        return candidates

    @staticmethod
    def _use_uploaded_pdf(document, uploaded):
        if not uploaded.uri:
            raise ValueError("Gemini PDF upload returned no file URI.")
        document.pop("data")
        document["uri"] = uploaded.uri

    def _prepare_pdf_uploads(self, kwargs):
        kwargs = copy.deepcopy(kwargs)
        for document in self._pdf_upload_candidates(kwargs):
            with io.BytesIO(base64.b64decode(document["data"])) as stream:
                uploaded = self.client.files.upload(file=stream, config={"mime_type": "application/pdf"})
            self._use_uploaded_pdf(document, uploaded)
        return kwargs

    async def _prepare_pdf_uploads_async(self, kwargs):
        kwargs = copy.deepcopy(kwargs)
        for document in self._pdf_upload_candidates(kwargs):
            with io.BytesIO(base64.b64decode(document["data"])) as stream:
                uploaded = await self.async_client.files.upload(file=stream, config={"mime_type": "application/pdf"})
            self._use_uploaded_pdf(document, uploaded)
        return kwargs

    def _create_interaction(self, conversation, model_name, **kwargs):
        prepared = self._prepare_pdf_uploads(kwargs)
        try:
            return self.client.interactions.create(**prepared)
        except Exception as error:
            if "previous_interaction_id" not in kwargs or not missing_continuation(
                error, "previous_interaction_id", kwargs["previous_interaction_id"]
            ):
                raise
            replay = self._replay_interaction_kwargs(conversation, model_name, kwargs)
            return self.client.interactions.create(**self._prepare_pdf_uploads(replay))

    async def _create_interaction_async(self, conversation, model_name, **kwargs):
        prepared = await self._prepare_pdf_uploads_async(kwargs)
        try:
            return await self.async_client.interactions.create(**prepared)
        except Exception as error:
            if "previous_interaction_id" not in kwargs or not missing_continuation(
                error, "previous_interaction_id", kwargs["previous_interaction_id"]
            ):
                raise
            replay = self._replay_interaction_kwargs(conversation, model_name, kwargs)
            return await self.async_client.interactions.create(**await self._prepare_pdf_uploads_async(replay))

    def _append_interaction(self, conversation, message, interaction, model):
        state = conversation.continuation("google", model) or {}
        environment_id = getattr(interaction, "environment_id", None) or state.get("environment_id")
        conversation.messages.append(message)
        conversation.checkpoint("google", model, message.id, environment_id=environment_id)

    # ------------------------------------------------------------------
    # Tools / generation_config / response_format
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        functions: List[BaseTool | Callable],
        additional_parameters: AdditionalParameters,
    ) -> List[Dict]:
        tools: List[Dict] = []
        for func in functions or []:
            if isinstance(func, BaseTool):
                decl = func.to_params(provider="google")
            elif callable(func):
                decl = self._callable_to_json_schema(func)
            else:
                raise TypeError("func must be either a BaseTool instance or a callable function")
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
        """Build the Interactions JSON response format without altering its schema."""
        structured_output = additional_parameters.get("structured_output")
        if structured_output is None or structured_output is False:
            return None
        if isinstance(structured_output, dict):
            schema = copy.deepcopy(structured_output)
        elif hasattr(structured_output, "model_json_schema"):
            schema = structured_output.model_json_schema()
        else:
            schema = TypeAdapter(structured_output).json_schema()
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
            kwargs["response_format"] = response_format

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
                arguments_json = arguments if isinstance(arguments, str) else json.dumps(provider_dump(arguments))
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

        return Message(
            id=getattr(interaction, "id", None),
            provider="google", model=model_name,
            **google_metadata(interaction),
            provider_data={"steps": provider_dump(getattr(interaction, "steps", []) or []),
                           "environment_id": getattr(interaction, "environment_id", None)},
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
            function_to_call = next((f for f in functions if self._tool_name(f) == fc.name), None)
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
            function_to_call = next((f for f in functions if self._tool_name(f) == fc.name), None)
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

    def _deep_research_interaction_params(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters,
    ) -> Dict:
        interaction_params = {
            **self._continuation_input(the_conversation, model),
            "agent": model,
            "background": True,
            "store": True,
        }
        self._prepend_research_instructions(interaction_params["input"], the_conversation)

        agent_config = dict(additional_parameters.get("agent_config") or {})
        if agent_config:
            agent_config.setdefault("type", "deep-research")
            interaction_params["agent_config"] = agent_config

        return interaction_params

    @staticmethod
    def _prepend_research_instructions(input_items, conversation):
        if conversation.system_prompt:
            input_items.insert(0, {
                "type": "user_input",
                "content": [{"type": "text", "text": f"System instructions:\n{conversation.system_prompt}"}],
            })

    def _poll_deep_research_interaction(self, interaction, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        while getattr(interaction, "status", None) in {"queued", "in_progress"}:
            time.sleep(min(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS, self._check_deadline(deadline, interaction)))
            self._check_deadline(deadline, interaction)
            interaction = self.client.interactions.get(interaction.id)
        return interaction

    async def _poll_deep_research_interaction_async(self, interaction, deadline=None):
        """Async counterpart of `_poll_deep_research_interaction`."""
        if deadline is None:
            deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        while getattr(interaction, "status", None) in {"queued", "in_progress"}:
            await asyncio.sleep(min(self.DEEP_RESEARCH_POLL_INTERVAL_SECONDS, self._check_deadline(deadline, interaction)))
            self._check_deadline(deadline, interaction)
            interaction = await self.async_client.interactions.get(interaction.id)
        return interaction

    def _parse_deep_research_interaction(self, interaction, model_name: str) -> Message:
        return self._parse_interaction_response(interaction, model_name)

    def _request_deep_research(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        additional_parameters: AdditionalParameters,
    ) -> Message:
        if functions:
            logger.warning("Gemini Deep Research agents do not support custom function tools.")

        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = self._create_interaction(
            the_conversation, model,
            **self._deep_research_interaction_params(model, the_conversation, additional_parameters)
        )
        interaction = self._poll_deep_research_interaction(interaction, deadline)
        assistant_message = self._parse_deep_research_interaction(
            interaction=interaction,
            model_name=model,
        )
        self._append_interaction(the_conversation, assistant_message, interaction, model)
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

        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = await self._create_interaction_async(
            the_conversation, model,
            **self._deep_research_interaction_params(model, the_conversation, additional_parameters)
        )
        interaction = await self._poll_deep_research_interaction_async(interaction, deadline)
        assistant_message = self._parse_deep_research_interaction(
            interaction=interaction,
            model_name=model,
        )
        self._append_interaction(the_conversation, assistant_message, interaction, model)
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
        if additional_parameters.get("new_environment"):
            the_conversation.reset_continuation("google", model)
        if agent_config := additional_parameters.get("agent_config"):
            kwargs["agent_config"] = {"type": "antigravity", **agent_config}
        if the_conversation.system_prompt:
            kwargs["system_instruction"] = the_conversation.system_prompt
        if tools := self._build_tools(functions, additional_parameters):
            kwargs["tools"] = tools
        return kwargs

    def _poll_agent_interaction(self, interaction, deadline=None):
        """Polls a background interaction until it reaches a terminal status or
        ``requires_action`` (waiting on a client-side function result)."""
        if deadline is None:
            deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        while getattr(interaction, "status", None) in {"queued", "in_progress"}:
            time.sleep(min(self.AGENT_POLL_INTERVAL_SECONDS, self._check_deadline(deadline, interaction)))
            self._check_deadline(deadline, interaction)
            interaction = self.client.interactions.get(interaction.id)
        return interaction

    async def _poll_agent_interaction_async(self, interaction, deadline=None):
        """Async counterpart of `_poll_agent_interaction`."""
        if deadline is None:
            deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        while getattr(interaction, "status", None) in {"queued", "in_progress"}:
            await asyncio.sleep(min(self.AGENT_POLL_INTERVAL_SECONDS, self._check_deadline(deadline, interaction)))
            self._check_deadline(deadline, interaction)
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

        custom_function_names = {self._tool_name(f) for f in functions}

        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = self._create_interaction(
            the_conversation, model,
            **self._continuation_input(the_conversation, model, antigravity=True),
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline, interaction)
            if background:
                interaction = self._poll_agent_interaction(interaction, deadline)

            assistant_message = self._parse_interaction_response(interaction, model)

            # Keep only the calls the platform is responsible for executing.
            custom_calls = [
                fc for fc in assistant_message.function_calls
                if fc.name in custom_function_names
            ]
            assistant_message.function_calls = custom_calls
            assistant_message.status = result_status(
                assistant_message.finish_reason, error=assistant_message.error, has_calls=bool(custom_calls),
            )
            assistant_message._replay_fingerprint = assistant_message._content_fingerprint()
            self._append_interaction(the_conversation, assistant_message, interaction, model)

            if not custom_calls or not assistant_message.can_execute_tools:
                return assistant_message

            self._check_deadline(deadline, interaction)
            function_responses = self._execute_function_calls(
                custom_calls, functions, tool_output_callback
            )
            the_conversation.messages.append(Message(role="function", content="", function_responses=function_responses))

            if _tool_round + 1 >= MAX_TOOL_ROUNDS:
                break
            self._check_deadline(deadline, interaction)
            interaction = self._create_interaction(
                the_conversation, model,
                input=[self._function_result_entry(fr) for fr in function_responses],
                previous_interaction_id=interaction.id,
                environment=self._continuation_input(the_conversation, model, antigravity=True)["environment"],
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

        custom_function_names = {self._tool_name(f) for f in functions}

        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = await self._create_interaction_async(
            the_conversation, model,
            **self._continuation_input(the_conversation, model, antigravity=True),
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline, interaction)
            if background:
                interaction = await self._poll_agent_interaction_async(interaction, deadline)

            assistant_message = self._parse_interaction_response(interaction, model)

            # Keep only the calls the platform is responsible for executing.
            custom_calls = [
                fc for fc in assistant_message.function_calls
                if fc.name in custom_function_names
            ]
            assistant_message.function_calls = custom_calls
            assistant_message.status = result_status(
                assistant_message.finish_reason, error=assistant_message.error, has_calls=bool(custom_calls),
            )
            assistant_message._replay_fingerprint = assistant_message._content_fingerprint()
            self._append_interaction(the_conversation, assistant_message, interaction, model)

            if not custom_calls or not assistant_message.can_execute_tools:
                return assistant_message

            self._check_deadline(deadline, interaction)
            function_responses = await self._execute_function_calls_async(
                custom_calls, functions, tool_output_callback
            )
            the_conversation.messages.append(Message(role="function", content="", function_responses=function_responses))

            if _tool_round + 1 >= MAX_TOOL_ROUNDS:
                break
            self._check_deadline(deadline, interaction)
            interaction = await self._create_interaction_async(
                the_conversation, model,
                input=[self._function_result_entry(fr) for fr in function_responses],
                previous_interaction_id=interaction.id,
                environment=self._continuation_input(the_conversation, model, antigravity=True)["environment"],
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

        # A valid provider/model checkpoint identifies the complete unsent suffix.
        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = self._create_interaction(
            the_conversation, model,
            **self._continuation_input(the_conversation, model),
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline, interaction)
            assistant_message = self._parse_interaction_response(interaction, model)
            self._append_interaction(the_conversation, assistant_message, interaction, model)

            if not assistant_message.function_calls or not assistant_message.can_execute_tools:
                return assistant_message

            # --- Execute tools and continue with previous_interaction_id ---
            self._check_deadline(deadline, interaction)
            function_responses = self._execute_function_calls(
                assistant_message.function_calls, functions, tool_output_callback
            )
            the_conversation.messages.append(Message(role="function", content="", function_responses=function_responses))

            result_inputs = [self._function_result_entry(fr) for fr in function_responses]
            if _tool_round + 1 >= MAX_TOOL_ROUNDS:
                break
            self._check_deadline(deadline, interaction)
            interaction = self._create_interaction(
                the_conversation, model,
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

        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        interaction = await self._create_interaction_async(
            the_conversation, model,
            **self._continuation_input(the_conversation, model),
            **base_kwargs,
        )

        for _tool_round in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline, interaction)
            assistant_message = self._parse_interaction_response(interaction, model)
            self._append_interaction(the_conversation, assistant_message, interaction, model)

            if not assistant_message.function_calls or not assistant_message.can_execute_tools:
                return assistant_message

            # --- Execute tools and continue with previous_interaction_id ---
            self._check_deadline(deadline, interaction)
            function_responses = await self._execute_function_calls_async(
                assistant_message.function_calls, functions, tool_output_callback
            )
            the_conversation.messages.append(Message(role="function", content="", function_responses=function_responses))

            result_inputs = [self._function_result_entry(fr) for fr in function_responses]
            if _tool_round + 1 >= MAX_TOOL_ROUNDS:
                break
            self._check_deadline(deadline, interaction)
            interaction = await self._create_interaction_async(
                the_conversation, model,
                input=result_inputs,
                previous_interaction_id=interaction.id,
                **base_kwargs,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
        )

