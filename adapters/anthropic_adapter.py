import inspect
from copy import deepcopy
import json
from loguru import logger
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import anthropic

from llm_platform.services.conversation import (
    Conversation,
    FunctionCall,
    FunctionResponse,
    Message,
    ThinkingResponse,
)
from llm_platform.services.files import (
    DocumentFile,
    ExcelDocumentFile,
    ImageFile,
    PDFDocumentFile,
    TextDocumentFile,
)
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import (
    function_call_to_anthropic,
    function_response_to_anthropic,
    provider_dump,
)

from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS, PDF_INLINE_MAX_BYTES, PDF_INLINE_MAX_PAGES
from llm_platform.types import AdditionalParameters
from .response_metadata import anthropic_metadata

# --- Constants ---

# Model-specific beta flags for experimental features
BETA_FLAGS = {
    "claude-sonnet-4-6": ["context-1m-2025-08-07"],
    "claude-opus-4-6": ["context-1m-2025-08-07"],
}

# Reasoning effort to 'thinking' token budget mapping
REASONING_BUDGETS = {
    "low": 4_000,
    "medium": 8_000,
    "high": 16_000,
}

# Buffer for response tokens when correcting max_tokens to avoid exceeding context window
RESPONSE_TOKEN_BUFFER = 1000

# Threshold for max_tokens above which streaming is required to avoid HTTP timeouts
MAX_TOKENS_STREAMING_THRESHOLD = 21_000


class ClaudeStreamProcessor:
    """
    Processes a stream of events from the Anthropic Messages API.

    This class accumulates data from various event types into a structured format,
    including thinking steps, final response text, tool usage, and token counts.
    """

    def __init__(self, model=None):
        self.model = model
        # Final outputs
        self.thinking_responses: List[ThinkingResponse] = []
        self.response_text: str = ""
        self.tool_uses: List[Dict] = []
        self.usage: Dict = {
            "model": "",
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        self.error = None
        self.stop_reason: str | None = None
        self.id = None
        self.container = None
        self.content_blocks = []
        self._native_blocks = {}
        self._native_tool_json = {}

        # Internal state for processing the stream
        self._current_thinking_text: str = ""
        self._current_tool_name: str | None = None
        self._current_tool_id: str | None = None
        self._current_tool_json: str = ""
        self._current_block_type: str | None = None
        self._current_block_signature: str | None = None

        # Event dispatcher
        self._event_handlers = {
            'message_start': self._handle_message_start,
            'content_block_start': self._handle_content_block_start,
            'content_block_delta': self._handle_content_block_delta,
            'content_block_stop': self._handle_content_block_stop,
            'message_delta': self._handle_message_delta,
        }

    def process_event(self, event: Any):
        """Process a single event from the Claude stream."""
        event_type = getattr(event, 'type', None)
        if event_type == 'error':
            self.error = provider_dump(event.error)
        if handler := self._event_handlers.get(event_type):
            handler(event)

        # Retain every block, including redacted thinking and hosted tools.
        index = getattr(event, "index", None)
        if event_type == "content_block_start":
            self._native_blocks[index] = provider_dump(event.content_block)
        elif event_type == "content_block_delta":
            block = self._native_blocks[index]
            delta = provider_dump(event.delta)
            kind = delta.pop("type")
            if kind == "input_json_delta":
                self._native_tool_json[index] = self._native_tool_json.get(index, "") + delta["partial_json"]
            elif kind == "citations_delta":
                block["citations"] = (block.get("citations") or []) + [delta["citation"]]
            else:
                for key, value in delta.items():
                    block[key] = block.get(key, "") + value if isinstance(value, str) else value
        elif event_type == "content_block_stop":
            block = self._native_blocks.pop(index)
            if index in self._native_tool_json:
                block["input"] = json.loads(self._native_tool_json.pop(index))
            self.content_blocks.append(block)

    def _handle_message_start(self, event: Any):
        message = getattr(event, 'message', None)
        if not message:
            return
        self.id = getattr(message, "id", None)
        self.container = provider_dump(getattr(message, "container", None))
        usage = message.usage
        self.usage["provider_usage"] = provider_dump(usage)
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        self.usage["model"] = message.model
        # With caching on, the API's input_tokens counts only the uncached remainder;
        # report prompt_tokens as the full input (uncached + cache read + cache write).
        self.usage["prompt_tokens"] = getattr(usage, 'input_tokens', 0) + cache_read + cache_creation
        self.usage["cache_read_tokens"] = cache_read
        self.usage["cache_creation_tokens"] = cache_creation

    def _handle_content_block_start(self, event: Any):
        content_block = getattr(event, 'content_block', None)
        if not content_block:
            return

        self._current_block_type = getattr(content_block, 'type', None)
        if self._current_block_type == 'tool_use':
            self._current_tool_name = getattr(content_block, 'name', None)
            self._current_tool_id = getattr(content_block, 'id', None)
            self._current_tool_json = ""

    def _handle_content_block_delta(self, event: Any):
        delta = getattr(event, 'delta', None)
        if not delta:
            return

        delta_type = getattr(delta, 'type', None)
        if delta_type == 'thinking_delta':
            self._current_thinking_text += getattr(delta, 'thinking', '')
        elif delta_type == 'signature_delta':
            self._current_block_signature = getattr(delta, 'signature', None)
        elif delta_type == 'text_delta':
            self.response_text += getattr(delta, 'text', '')
        elif delta_type == 'input_json_delta':
            self._current_tool_json += getattr(delta, 'partial_json', '')

    def _handle_content_block_stop(self, event: Any):
        if self._current_block_type == 'tool_use' and self._current_tool_name:
            try:
                parameters = json.loads(self._current_tool_json) if self._current_tool_json else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON for tool '{self._current_tool_name}'. Using empty parameters.")
                parameters = {}

            self.tool_uses.append({
                'name': self._current_tool_name,
                'id': self._current_tool_id,
                'parameters': parameters,
            })
            self._current_tool_name = None
            self._current_tool_id = None
            self._current_tool_json = ""

        elif self._current_block_type == 'thinking' and self._current_thinking_text:
            self.thinking_responses.append(
                ThinkingResponse(content=self._current_thinking_text, id=self._current_block_signature)
            )
            self._current_thinking_text = ""
            self._current_block_signature = None

        self._current_block_type = None

    def _handle_message_delta(self, event: Any):
        self.usage.setdefault("provider_usage", {}).update(provider_dump(event.usage) or {})
        self.usage["completion_tokens"] = getattr(event.usage, 'output_tokens', 0)
        self.stop_reason = getattr(event.delta, 'stop_reason', None) or self.stop_reason
        if getattr(event.delta, "container", None) is not None:
            self.container = provider_dump(event.delta.container)


class AnthropicAdapter(AdapterBase):
    """Adapter for interacting with the Anthropic Claude API."""

    def __init__(self):
        super().__init__()
        self._async_client = None

    def _build_client(self):
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @property
    def async_client(self):
        """Async SDK client, constructed lazily once on first access."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._async_client

    # --- Main Public Methods ---

    def _build_tools(self, functions, parameters):
        tools = [self._convert_function_to_tool(func) for func in functions or []]
        if parameters.get("web_search"):
            options = deepcopy(parameters.get("web_search_options") or {})
            allowed = {"max_uses", "allowed_domains", "blocked_domains", "user_location",
                       "allowed_callers", "response_inclusion"}
            if not isinstance(options, dict) or options.keys() - allowed:
                raise ValueError("Unsupported Claude web_search_options")
            if "allowed_domains" in options and "blocked_domains" in options:
                raise ValueError("Use allowed_domains or blocked_domains, not both")
            if "max_uses" in options and (type(options["max_uses"]) is not int or options["max_uses"] < 1):
                raise ValueError("web_search_options.max_uses must be a positive integer")
            if options.get("response_inclusion", "full") not in {"full", "excluded"}:
                raise ValueError("web_search_options.response_inclusion must be full or excluded")
            tools.append({"type": "web_search_20260318", "name": "web_search",
                          "max_uses": 10, **options})
        if parameters.get("code_execution"):
            tools.append({"type": "code_execution_20260521", "name": "code_execution"})
        if parameters.get("web_search_options") and not parameters.get("web_search"):
            raise ValueError("web_search_options requires web_search=True")
        return tools

    def _message_from_processor(self, processor, **kwargs):
        kwargs.setdefault("function_calls", [
            FunctionCall(id=call["id"], name=call["name"], arguments=call["parameters"])
            for call in processor.tool_uses
        ])
        return Message(
            role="assistant", content=processor.response_text,
            id=processor.id, provider="anthropic", model=processor.model or processor.usage["model"],
            provider_data={"content": processor.content_blocks, "container": processor.container},
            thinking_responses=processor.thinking_responses, usage=processor.usage,
            **anthropic_metadata(processor), **kwargs,
        )

    def _round_kwargs(self, model, conversation, parameters, tools, **kwargs):
        request = self._prepare_request_kwargs(model, parameters, **kwargs)
        if container_id := self._continuation_container(conversation, model):
            request.setdefault("container", container_id)
        return {
            "model": model, "system": conversation.system_prompt,
            "messages": self.convert_conversation_history_to_adapter_format(conversation, parameters, model),
            "tools": tools, **request,
        }

    def request_llm(
        self, model: str, the_conversation: Conversation,
        functions: List[BaseTool | Callable] | None = None,
        tool_output_callback: Callable | None = None,
        additional_parameters: AdditionalParameters | None = None, **kwargs,
    ) -> Message:
        """Run local tools and resume hosted pauses within one bounded turn."""
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        tools = self._build_tools(functions, parameters)
        stream = (parameters.get("max_tokens") or 0) >= MAX_TOKENS_STREAMING_THRESHOLD
        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        for _ in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline)
            request = self._round_kwargs(model, the_conversation, parameters, tools, **kwargs)
            if "max_tokens" in request:
                request["max_tokens"] = self.correct_max_tokens(model, request["messages"], request["max_tokens"], tools)
            if stream:
                events = self.client.beta.messages.create(**request, stream=True)
                processor = ClaudeStreamProcessor(model)
                try:
                    for event in events:
                        processor.process_event(event)
                        self._check_deadline(deadline, processor)
                finally:
                    events.close()
            else:
                response = self.client.beta.messages.create(**request)
                processor = self._parse_non_streaming_response(response, model)
            message = self._message_from_processor(processor)
            if message.status == "requires_action" and processor.tool_uses:
                self._check_deadline(deadline, processor)
                self._handle_tool_calls(processor, the_conversation, functions or [], tool_output_callback)
            else:
                the_conversation.messages.append(message)
                if message.status != "paused":
                    return message
            self._check_deadline(deadline, processor)
        raise RuntimeError(f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}")

    async def request_llm_async(
        self, model: str, the_conversation: Conversation,
        functions: List[BaseTool | Callable] | None = None,
        tool_output_callback: Callable | None = None,
        additional_parameters: AdditionalParameters | None = None, **kwargs,
    ) -> Message:
        """Native async equivalent, including streamed pauses and coroutine tools."""
        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        tools = self._build_tools(functions, parameters)
        stream = (parameters.get("max_tokens") or 0) >= MAX_TOKENS_STREAMING_THRESHOLD
        deadline = time.monotonic() + self.RESPONSE_TIMEOUT_SECONDS
        for _ in range(MAX_TOOL_ROUNDS):
            self._check_deadline(deadline)
            request = self._round_kwargs(model, the_conversation, parameters, tools, **kwargs)
            if "max_tokens" in request:
                request["max_tokens"] = await self.correct_max_tokens_async(model, request["messages"], request["max_tokens"], tools)
            if stream:
                events = await self.async_client.beta.messages.create(**request, stream=True)
                processor = ClaudeStreamProcessor(model)
                try:
                    async for event in events:
                        processor.process_event(event)
                        self._check_deadline(deadline, processor)
                finally:
                    await events.close()
            else:
                response = await self.async_client.beta.messages.create(**request)
                processor = self._parse_non_streaming_response(response, model)
            message = self._message_from_processor(processor)
            if message.status == "requires_action" and processor.tool_uses:
                self._check_deadline(deadline, processor)
                await self._handle_tool_calls_async(processor, the_conversation, functions or [], tool_output_callback)
            else:
                the_conversation.messages.append(message)
                if message.status != "paused":
                    return message
            self._check_deadline(deadline, processor)
        raise RuntimeError(f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}")

    def _parse_non_streaming_response(self, response, model=None) -> ClaudeStreamProcessor:
        """Converts a non-streaming API response into a ClaudeStreamProcessor for uniform handling."""
        processor = ClaudeStreamProcessor(model)
        processor.id = getattr(response, "id", None)
        processor.error = provider_dump(getattr(response, "error", None))
        processor.container = provider_dump(getattr(response, "container", None))
        processor.content_blocks = provider_dump(response.content)
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        processor.usage["model"] = response.model
        # With caching on, the API's input_tokens counts only the uncached remainder;
        # report prompt_tokens as the full input (uncached + cache read + cache write).
        processor.usage["prompt_tokens"] = response.usage.input_tokens + cache_read + cache_creation
        processor.usage["completion_tokens"] = response.usage.output_tokens
        processor.usage["cache_read_tokens"] = cache_read
        processor.usage["cache_creation_tokens"] = cache_creation
        processor.usage["provider_usage"] = provider_dump(response.usage)
        processor.stop_reason = response.stop_reason

        for block in response.content:
            if block.type == "thinking":
                processor.thinking_responses.append(
                    ThinkingResponse(content=block.thinking, id=block.signature)
                )
            elif block.type == "text":
                processor.response_text += block.text
            elif block.type == "tool_use":
                processor.tool_uses.append({
                    'name': block.name,
                    'id': block.id,
                    'parameters': block.input,
                })

        return processor

    def _handle_tool_calls(
        self,
        processor: ClaudeStreamProcessor,
        conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
    ):
        """Executes tool calls requested by the model and updates the conversation."""
        exchanges = []
        for tool_call in processor.tool_uses:
            function_to_call = self._find_tool_function(tool_call["name"], functions)
            parameters = tool_call.get("parameters", {})
            if function_to_call is None:
                response = self._missing_tool_response(tool_call["name"])
            else:
                # Execute the function with keyword arguments for robustness.
                response = function_to_call(**parameters)

            if tool_output_callback:
                tool_output_callback(tool_call['name'], parameters, response)
            exchanges.append((tool_call, parameters, response))

        self._append_tool_exchange_message(processor, conversation, exchanges)

    async def _handle_tool_calls_async(
        self,
        processor: ClaudeStreamProcessor,
        conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
    ):
        """Async counterpart of `_handle_tool_calls`; additionally supports coroutine tools."""
        exchanges = []
        for tool_call in processor.tool_uses:
            function_to_call = self._find_tool_function(tool_call["name"], functions)
            parameters = tool_call.get("parameters", {})
            if function_to_call is None:
                response = self._missing_tool_response(tool_call["name"])
            elif inspect.iscoroutinefunction(function_to_call):
                response = await function_to_call(**parameters)
            else:
                response = function_to_call(**parameters)

            if tool_output_callback:
                tool_output_callback(tool_call['name'], parameters, response)
            exchanges.append((tool_call, parameters, response))

        self._append_tool_exchange_message(processor, conversation, exchanges)

    def _find_tool_function(self, name: str, functions: List[BaseTool]) -> BaseTool | Callable | None:
        """Finds the local function for a tool call by its name (BaseTool exposes
        `.name`, plain callables `__name__`)."""
        return next((f for f in functions if self._tool_name(f) == name), None)

    def _missing_tool_response(self, name: str) -> Dict:
        """Error result for a tool call whose function was not provided.

        Every tool_use must get a tool_result: report the failure to the model
        instead of dropping the call, which would make it retry forever.
        """
        logger.error(f"Function '{name}' not found in provided tools.")
        return {"error": f"Tool '{name}' not found in provided tools."}

    def _append_tool_exchange_message(
        self,
        processor: ClaudeStreamProcessor,
        conversation: Conversation,
        exchanges: List[Tuple[Dict, Dict, Any]],
    ):
        """Adds the assistant's message (requesting the tool use) and the tool results to the conversation."""
        function_calls = []
        function_responses = []
        for tool_call, parameters, response in exchanges:
            function_calls.append(FunctionCall(id=tool_call['id'], name=tool_call['name'], arguments=parameters))
            function_responses.append(FunctionResponse(id=tool_call['id'], name=tool_call['name'], response=response))

        assistant_message = self._message_from_processor(
            processor, function_calls=function_calls, function_responses=function_responses,
        )
        conversation.messages.append(assistant_message)

    @staticmethod
    def _continuation_container(conversation, model):
        for message in reversed(conversation.messages):
            container = message.replay_data("anthropic", model).get("container")
            if container:
                return container.get("id") if isinstance(container, dict) else container
        return None

    def _prepare_request_kwargs(
        self,
        model: str,
        additional_parameters: AdditionalParameters,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepares kwargs for the API call, handling reasoning, betas, etc."""
        request_kwargs: Dict[str, Any] = {}
        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            request_kwargs.update(kwargs)

        if max_tokens := additional_parameters.get("max_tokens", None):
            request_kwargs["max_tokens"] = max_tokens

        temperature = additional_parameters.get("temperature", None)
        if temperature is not None:
            request_kwargs['temperature'] = temperature

        output_config = {}

        # Newer models (Opus 4.7/4.8, Sonnet 4.6, ...) require adaptive thinking:
        # `thinking.type: "enabled"` with a fixed budget returns a 400 on them
        # ("use thinking.type.adaptive and output_config.effort"). This is driven by
        # the per-model `adaptive_thinking` flag in models_config.yaml, so enabling a
        # new model is a config change rather than editing a hardcoded model list.
        model_object = self.model_config[model]
        if model_object and model_object["adaptive_thinking"]:
            reasoning_effort = additional_parameters.get("reasoning", {}).get("effort", None)
            if reasoning_effort:
                request_kwargs['thinking'] = {"type": "adaptive"}
                output_config["effort"] = reasoning_effort
        else:
            reasoning_effort = additional_parameters.get("reasoning", {}).get("effort", "none")
            if budget_tokens := REASONING_BUDGETS.get(reasoning_effort):
                request_kwargs['thinking'] = {"type": "enabled", "budget_tokens": budget_tokens}

        # Structured output — convert Pydantic model to JSON schema for output_config.format
        if structured_output_class := additional_parameters.get("structured_output", None):
            if hasattr(structured_output_class, "model_json_schema"):
                schema = structured_output_class.model_json_schema()
            else:
                from pydantic import TypeAdapter
                schema = TypeAdapter(structured_output_class).json_schema()
            schema = anthropic.transform_schema(schema)
            output_config["format"] = {"type": "json_schema", "schema": schema}

        if output_config:
            request_kwargs["output_config"] = output_config

        if beta_flag := BETA_FLAGS.get(model):
            request_kwargs['betas'] = beta_flag

        # Enable automatic prompt caching. A single top-level cache_control places the
        # breakpoint on the last cacheable block and moves it forward as the conversation
        # grows, so the stable system + tools + history prefix is served from cache on
        # subsequent turns and tool-use loops. Prompts below the model's minimum cacheable
        # length are silently left uncached, so this is safe to apply unconditionally.
        request_kwargs["cache_control"] = {"type": "ephemeral"}

        return request_kwargs

    # --- Conversation and Tool Formatting ---

    def convert_conversation_history_to_adapter_format(
        self, conversation: Conversation, additional_parameters: AdditionalParameters | None = None,
        model: str | None = None,
    ) -> List[Dict]:
        """Converts a Conversation object into the format required by the Anthropic API."""
        if additional_parameters is None:
            additional_parameters = {}

        history = []
        citations_enabled = additional_parameters.get(
            "citations_enabled",
            False,
        )

        for message in conversation.messages:
            if message.role == "function":
                history.append({"role": "user", "content": [
                    function_response_to_anthropic(fr) for fr in message.function_responses
                ]})
                continue
            native = message.replay_data("anthropic", model)
            content = native.get("content")
            if content is None:
                content = self._prepare_message_content(message, citations_enabled)
            history.append({"role": message.role, "content": content})

            # Per Anthropic's API, tool results must be in a separate, subsequent user message.
            if message.function_responses:
                tool_results_content = [function_response_to_anthropic(fr) for fr in message.function_responses]
                history.append({"role": "user", "content": tool_results_content})

        return history

    def _prepare_message_content(self, message: Message, citations_enabled: bool) -> List[Dict]:
        """Builds the 'content' list for a single message, handling text, files, and tool calls."""
        content_list = self._ensure_content_is_list(message.content)

        if message.files:
            # Documents should be prepended to the content list.
            for file in message.files:
                if isinstance(file, DocumentFile):
                    content_list.insert(0, self._format_document_content(file, citations_enabled))

            # Images are appended.
            for file in message.files:
                if isinstance(file, ImageFile):
                    content_list.append(self._format_image_content(file))
                elif not isinstance(file, DocumentFile):
                    raise ValueError(f"Unsupported file type for Anthropic: {type(file).__name__}")

        # For assistant messages, add thinking and tool call requests.
        if message.role == "assistant":
            if message.function_calls:
                tool_call_content = [function_call_to_anthropic(fc) for fc in message.function_calls]
                content_list.extend(tool_call_content)

        return content_list

    def _ensure_content_is_list(self, content: Any) -> List[Dict]:
        """Ensures message content is a list, converting a string if necessary."""
        if isinstance(content, list):
            return content
        text = content if content and str(content).strip() else " "
        return [{"type": "text", "text": text}]

    def _format_image_content(self, file: ImageFile) -> Dict:
        """Formats an ImageFile into an Anthropic content block."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": file.mime_type,
                "data": file.base64,
            },
        }

    def _format_document_content(self, file: DocumentFile, citations_enabled: bool) -> Dict:
        """Formats a DocumentFile into an Anthropic content block."""
        # For small PDFs, upload the raw file. For large PDFs or other doc types, extract text.
        if isinstance(file, PDFDocumentFile) and file.size < PDF_INLINE_MAX_BYTES and file.number_of_pages < PDF_INLINE_MAX_PAGES:
            source = {"type": "base64", "media_type": "application/pdf", "data": file.base64}
        else:
            source = {"type": "text", "media_type": "text/plain", "data": file.text}

        return {
            "type": "document",
            "source": source,
            "title": file.name,
            "context": "This is a trustworthy document.",
            "citations": {"enabled": citations_enabled},
        }

    def _convert_function_to_tool(self, func: BaseTool | Callable) -> Dict:
        """Converts a BaseTool or a standard Python function into an Anthropic tool definition."""
        if isinstance(func, BaseTool):
            return func.to_params(provider='anthropic')
        if callable(func):
            return self._convert_callable_to_tool(func)
        raise TypeError("Tool must be a BaseTool instance or a callable function.")

    def _convert_callable_to_tool(self, func: Callable) -> Dict:
        """Uses introspection to create a tool definition from a Python function."""
        schema = self._callable_to_json_schema(func)
        return {
            'name': schema['name'],
            'description': schema['description'],
            'input_schema': schema['parameters'],
        }

    # --- Token Counting ---

    def count_tokens(self, model: str, messages: List[Dict], tools: List[Dict] | None = None) -> int:
        """Counts the number of input tokens for a given model, message list, and tools."""
        try:
            response = self.client.messages.count_tokens(model=model, messages=messages, tools=tools or [])
            return response.input_tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for model {model}: {e}")
            return 0

    async def count_tokens_async(self, model: str, messages: List[Dict], tools: List[Dict] | None = None) -> int:
        """Async counterpart of `count_tokens`."""
        try:
            response = await self.async_client.messages.count_tokens(model=model, messages=messages, tools=tools or [])
            return response.input_tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for model {model}: {e}")
            return 0

    def correct_max_tokens(self, model: str, messages: List[Dict], max_tokens: int, tools: List[Dict] | None = None) -> int:
        """Adjusts max_tokens to prevent exceeding the model's context window."""
        request_tokens = self.count_tokens(model, messages, tools)
        return self._clamp_max_tokens(model, request_tokens, max_tokens)

    async def correct_max_tokens_async(self, model: str, messages: List[Dict], max_tokens: int, tools: List[Dict] | None = None) -> int:
        """Async counterpart of `correct_max_tokens`."""
        request_tokens = await self.count_tokens_async(model, messages, tools)
        return self._clamp_max_tokens(model, request_tokens, max_tokens)

    def _clamp_max_tokens(self, model: str, request_tokens: int, max_tokens: int) -> int:
        """Clamps max_tokens against the model's own limit and remaining context window."""
        specific_model_object = self.model_config[model]
        context_window = specific_model_object.context_window

        if max_tokens is None:
            max_tokens = specific_model_object.max_tokens
        elif max_tokens > specific_model_object.max_tokens:
            logger.warning(
                f"Requested max_tokens ({max_tokens}) exceeds model's max_tokens ({specific_model_object.max_tokens}). "
                f"Correcting to {specific_model_object.max_tokens}."
            )
            max_tokens = specific_model_object.max_tokens

        if request_tokens + max_tokens >= context_window:
            new_max_tokens = context_window - request_tokens - RESPONSE_TOKEN_BUFFER
            if new_max_tokens < 0:
                new_max_tokens = 0 # Cannot have negative tokens
            logger.warning(
                f"Request tokens ({request_tokens}) + max_tokens ({max_tokens}) exceeds context window "
                f"({context_window}). Correcting max_tokens to {new_max_tokens}."
            )
            return new_max_tokens
        return max_tokens
