import asyncio
import json
from loguru import logger
import os
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
    thinking_response_to_anthropic,
)

from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS, PDF_INLINE_MAX_BYTES, PDF_INLINE_MAX_PAGES
from llm_platform.types import AdditionalParameters

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

    def __init__(self):
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
        self.stop_reason: str | None = None

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
        if handler := self._event_handlers.get(event_type):
            handler(event)

    def _handle_message_start(self, event: Any):
        message = getattr(event, 'message', None)
        if not message:
            return
        usage = message.usage
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
        self.usage["completion_tokens"] = getattr(event.usage, 'output_tokens', 0)
        self.stop_reason = getattr(event.delta, 'stop_reason', None)


class AnthropicAdapter(AdapterBase):
    """Adapter for interacting with the Anthropic Claude API."""

    def _build_client(self):
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # --- Main Public Methods ---

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
        Sends a request to the LLM, handling simple responses and tool use.

        Uses streaming for requests with max_tokens >= MAX_TOKENS_STREAMING_THRESHOLD
        to avoid HTTP timeouts. Uses non-streaming otherwise, which enables structured output.
        """
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        max_tokens = additional_parameters.get("max_tokens") or 0
        use_streaming = max_tokens >= MAX_TOKENS_STREAMING_THRESHOLD

        if use_streaming and additional_parameters.get("structured_output", None):
            # Structured output is only supported on the non-streaming path. Cap
            # max_tokens below the streaming threshold instead of failing, so the
            # config-default max_tokens (64k/128k) still works with structured output.
            capped_max_tokens = MAX_TOKENS_STREAMING_THRESHOLD - RESPONSE_TOKEN_BUFFER
            logger.warning(
                f"structured_output requires a non-streaming request; capping max_tokens "
                f"from {max_tokens} to {capped_max_tokens}."
            )
            additional_parameters["max_tokens"] = capped_max_tokens
            use_streaming = False

        if use_streaming:
            if functions:
                processor = self._request_llm_with_tools_streaming(
                    model=model,
                    conversation=the_conversation,
                    functions=functions,
                    tool_output_callback=tool_output_callback,
                    additional_parameters=additional_parameters,
                    **kwargs,
                )
            else:
                processor = self._request_llm_simple_streaming(
                    model=model,
                    conversation=the_conversation,
                    additional_parameters=additional_parameters,
                    **kwargs,
                )
        else:
            if functions:
                processor = self._request_llm_with_tools(
                    model=model,
                    conversation=the_conversation,
                    functions=functions,
                    tool_output_callback=tool_output_callback,
                    additional_parameters=additional_parameters,
                    **kwargs,
                )
            else:
                processor = self._request_llm_simple(
                    model=model,
                    conversation=the_conversation,
                    additional_parameters=additional_parameters,
                    **kwargs,
                )

        response_message = Message(
            role="assistant",
            content=processor.response_text,
            thinking_responses=processor.thinking_responses,
            usage=processor.usage,
        )
        the_conversation.messages.append(response_message)
        return response_message

    # --- Private Helper Methods for LLM Requests ---

    def _parse_non_streaming_response(self, response) -> ClaudeStreamProcessor:
        """Converts a non-streaming API response into a ClaudeStreamProcessor for uniform handling."""
        processor = ClaudeStreamProcessor()
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        processor.usage["model"] = response.model
        # With caching on, the API's input_tokens counts only the uncached remainder;
        # report prompt_tokens as the full input (uncached + cache read + cache write).
        processor.usage["prompt_tokens"] = response.usage.input_tokens + cache_read + cache_creation
        processor.usage["completion_tokens"] = response.usage.output_tokens
        processor.usage["cache_read_tokens"] = cache_read
        processor.usage["cache_creation_tokens"] = cache_creation
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

    def _request_llm_simple(
        self,
        model: str,
        conversation: Conversation,
        additional_parameters: AdditionalParameters,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles a non-tool-use, non-streaming request."""
        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        request_kwargs = self._prepare_request_kwargs(model, additional_parameters, **kwargs)

        if 'max_tokens' in request_kwargs:
            request_kwargs['max_tokens'] = self.correct_max_tokens(model, history, request_kwargs['max_tokens'])

        tools = []
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})
        if additional_parameters.get("code_execution", False):
            tools.append({"type": "code_execution_20250825", "name": "code_execution"})

        response = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            tools=tools,
            **request_kwargs,
        )

        return self._parse_non_streaming_response(response)

    def _request_llm_with_tools(
        self,
        model: str,
        conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
        additional_parameters: AdditionalParameters,
        _tool_round: int = 0,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles the recursive, non-streaming tool-use loop."""
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        tools = [self._convert_function_to_tool(func) for func in functions]
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

        request_kwargs = self._prepare_request_kwargs(model, additional_parameters, **kwargs)

        response = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            tools=tools,
            **request_kwargs,
        )

        processor = self._parse_non_streaming_response(response)

        if processor.stop_reason == "tool_use":
            self._handle_tool_calls(processor, conversation, functions, tool_output_callback)
            return self._request_llm_with_tools(
                model, conversation, functions, tool_output_callback, additional_parameters,
                _tool_round=_tool_round + 1, **kwargs
            )

        return processor

    def _request_llm_simple_streaming(
        self,
        model: str,
        conversation: Conversation,
        additional_parameters: AdditionalParameters,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles a non-tool-use streaming request."""
        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        request_kwargs = self._prepare_request_kwargs(model, additional_parameters, **kwargs)

        if 'max_tokens' in request_kwargs:
            request_kwargs['max_tokens'] = self.correct_max_tokens(model, history, request_kwargs['max_tokens'])

        tools = []
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})
        if additional_parameters.get("code_execution", False):
            tools.append({"type": "code_execution_20250825", "name": "code_execution"})

        stream = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            stream=True,
            tools=tools,
            **request_kwargs,
        )

        processor = ClaudeStreamProcessor()
        for event in stream:
            processor.process_event(event)
        return processor

    def _request_llm_with_tools_streaming(
        self,
        model: str,
        conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
        additional_parameters: AdditionalParameters,
        _tool_round: int = 0,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles the recursive, streaming tool-use loop."""
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        tools = [self._convert_function_to_tool(func) for func in functions]
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

        request_kwargs = self._prepare_request_kwargs(model, additional_parameters, **kwargs)

        stream = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            tools=tools,
            stream=True,
            **request_kwargs,
        )

        processor = ClaudeStreamProcessor()
        for event in stream:
            processor.process_event(event)

        # If the model wants to use a tool, handle the tool call cycle.
        if processor.stop_reason == "tool_use":
            self._handle_tool_calls(processor, conversation, functions, tool_output_callback)
            # Recursively call to get the final response after tool execution.
            return self._request_llm_with_tools_streaming(
                model, conversation, functions, tool_output_callback, additional_parameters,
                _tool_round=_tool_round + 1, **kwargs
            )

        # If no tool use, return the final processor state.
        return processor

    def _handle_tool_calls(
        self,
        processor: ClaudeStreamProcessor,
        conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable,
    ):
        """Executes tool calls requested by the model and updates the conversation."""
        function_calls = []
        function_responses = []

        for tool_call in processor.tool_uses:
            # Find the function to execute by its name (BaseTool exposes `.name`,
            # plain callables `__name__`).
            function_to_call = next(
                (f for f in functions if self._tool_name(f) == tool_call["name"]), None
            )

            parameters = tool_call.get("parameters", {})
            if function_to_call is None:
                # Every tool_use must get a tool_result: report the failure to the
                # model instead of dropping the call, which would make it retry forever.
                logger.error(f"Function '{tool_call['name']}' not found in provided tools.")
                response = {"error": f"Tool '{tool_call['name']}' not found in provided tools."}
            else:
                # Execute the function with keyword arguments for robustness.
                response = function_to_call(**parameters)

            # Record the call and response for the conversation history.
            function_calls.append(FunctionCall(id=tool_call['id'], name=tool_call['name'], arguments=parameters))
            function_responses.append(FunctionResponse(id=tool_call['id'], name=tool_call['name'], response=response))

            if tool_output_callback:
                tool_output_callback(tool_call['name'], parameters, response)

        # Add the assistant's message (requesting the tool use) and the tool results to the conversation.
        assistant_message = Message(
            role="assistant",
            content=processor.response_text,
            thinking_responses=processor.thinking_responses,
            function_calls=function_calls,
            function_responses=function_responses,
            usage=processor.usage,
        )
        conversation.messages.append(assistant_message)

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
        self, conversation: Conversation, additional_parameters: AdditionalParameters | None = None
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
            if message.thinking_responses:
                thinking_content = [thinking_response_to_anthropic(tr) for tr in message.thinking_responses]
                content_list = thinking_content + content_list  # Prepend thinking blocks

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
                "media_type": f"image/{file.extension}",
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
        if tools is None:
            tools = []
        try:
            response = self.client.messages.count_tokens(model=model, messages=messages, tools=tools)
            return response.input_tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for model {model}: {e}")
            return 0

    def correct_max_tokens(self, model: str, messages: List[Dict], max_tokens: int, tools: List[Dict] | None = None) -> int:
        """Adjusts max_tokens to prevent exceeding the model's context window."""
        if tools is None:
            tools = []
        request_tokens = self.count_tokens(model, messages, tools)
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
