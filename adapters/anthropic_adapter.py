import asyncio
import inspect
import json
from loguru import logger
import os
from typing import Any, Callable, Dict, List, Tuple

import anthropic

from llm_platform.helpers.model_config import ModelConfig
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

from .adapter_base import AdapterBase

# --- Constants ---

# Model-specific beta flags for experimental features
BETA_FLAGS = {
    "claude-sonnet-4-5": ["context-1m-2025-08-07"],
}

# Reasoning effort to 'thinking' token budget mapping
REASONING_BUDGETS = {
    "low": 4_000,
    "medium": 8_000,
    "high": 16_000,
}

# Buffer for response tokens when correcting max_tokens to avoid exceeding context window
RESPONSE_TOKEN_BUFFER = 1000

# Python type to JSON schema type mapping for tool generation
PYTHON_TYPE_TO_JSON_SCHEMA = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
}
DEFAULT_JSON_SCHEMA_TYPE = 'string'


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
        self.usage: Dict = {"model": "", "completion_tokens": 0, "prompt_tokens": 0}
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

    def process_event(self, event: any):
        """Process a single event from the Claude stream."""
        event_type = getattr(event, 'type', None)
        if handler := self._event_handlers.get(event_type):
            handler(event)

    def _handle_message_start(self, event: any):
        message = getattr(event, 'message', None)
        if not message:
            return
        self.usage["model"] = message.model
        self.usage["prompt_tokens"] = getattr(message.usage, 'input_tokens', 0)

    def _handle_content_block_start(self, event: any):
        content_block = getattr(event, 'content_block', None)
        if not content_block:
            return

        self._current_block_type = getattr(content_block, 'type', None)
        if self._current_block_type == 'tool_use':
            self._current_tool_name = getattr(content_block, 'name', None)
            self._current_tool_id = getattr(content_block, 'id', None)
            self._current_tool_json = ""

    def _handle_content_block_delta(self, event: any):
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

    def _handle_content_block_stop(self, event: any):
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

    def _handle_message_delta(self, event: any):
        self.usage["completion_tokens"] = getattr(event.usage, 'output_tokens', 0)
        self.stop_reason = getattr(event.delta, 'stop_reason', None)


class AnthropicAdapter(AdapterBase):
    """Adapter for interacting with the Anthropic Claude API."""

    def __init__(self):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_config = ModelConfig()

    # --- Main Public Methods ---

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        temperature: int = 0,
        tool_output_callback: Callable = None,
        additional_parameters: Dict = None,
        **kwargs,
    ) -> Message:
        """
        Sends a request to the LLM, handling simple responses and tool use.
        """
        if additional_parameters is None:
            additional_parameters = {}

        if temperature not in (None, 0) and "temperature" not in additional_parameters:
            additional_parameters["temperature"] = temperature

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)

        if functions:
            processor = self._request_llm_with_tools_streaming(
                model=model,
                conversation=the_conversation,
                functions=functions,
                temperature=temperature,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
                **kwargs,
            )
        else:
            processor = self._request_llm_simple_streaming(
                model=model,
                conversation=the_conversation,
                temperature=temperature,
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

    def voice_to_text(self, audio_file):
        raise NotImplementedError("Anthropic does not support voice-to-text functionality.")

    def generate_image(self, prompt: str, size: str, quality: str, n=1):
        raise NotImplementedError("Anthropic does not support image generation.")

    def get_models(self) -> List[str]:
        """
        Returns a list of known Anthropic models.
        Note: Anthropic does not provide a public API endpoint to list models,
        so this list is maintained manually.
        """
        raise NotImplementedError("Anthropic does not provide a public API to list models. Use known models instead.")

    # --- Private Helper Methods for LLM Requests ---

    def _request_llm_simple_streaming(
        self,
        model: str,
        conversation: Conversation,
        temperature: int,
        additional_parameters: Dict,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles a non-tool-use streaming request."""
        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        request_kwargs, temp = self._prepare_request_kwargs(model, temperature, additional_parameters, **kwargs)

        if 'max_tokens' in request_kwargs:
            request_kwargs['max_tokens'] = self.correct_max_tokens(model, history, request_kwargs['max_tokens'])

        tools = []
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

        stream = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            temperature=temp,
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
        temperature: int,
        tool_output_callback: Callable,
        additional_parameters: Dict,
        **kwargs,
    ) -> ClaudeStreamProcessor:
        """Handles the recursive, streaming tool-use loop."""
        history = self.convert_conversation_history_to_adapter_format(conversation, additional_parameters)
        tools = [self._convert_function_to_tool(func) for func in functions]
        if additional_parameters.get("web_search", False):
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

        request_kwargs, temp = self._prepare_request_kwargs(model, temperature, additional_parameters, **kwargs)

        stream = self.client.beta.messages.create(
            model=model,
            system=conversation.system_prompt,
            messages=history,
            temperature=temp,
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
                model, conversation, functions, temperature, tool_output_callback, additional_parameters, **kwargs
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
            try:
                # Find the function to execute by its name.
                function_to_call = next(f for f in functions if f.name == tool_call["name"])
            except StopIteration:
                logger.error(f"Function '{tool_call['name']}' not found in provided tools.")
                continue

            # Execute the function with keyword arguments for robustness.
            parameters = tool_call.get("parameters", {})
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
        temperature: int,
        additional_parameters: Dict,
        **kwargs,
    ) -> Tuple[Dict, int]:
        """Prepares kwargs for the API call, handling reasoning, betas, etc."""
        request_kwargs: Dict[str, Any] = {}
        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            request_kwargs.update(kwargs)

        if (max_tokens := additional_parameters.get("max_tokens")) is not None:
            request_kwargs["max_tokens"] = max_tokens

        temp = additional_parameters.get("temperature", temperature)

        reasoning_effort = additional_parameters.get("reasoning", {}).get("effort", "none")
        if budget_tokens := REASONING_BUDGETS.get(reasoning_effort):
            request_kwargs['thinking'] = {"type": "enabled", "budget_tokens": budget_tokens}
            temp = 1  # Anthropic recommends temp=1 for best results with thinking

        if beta_flag := BETA_FLAGS.get(model):
            request_kwargs['betas'] = beta_flag

        return request_kwargs, temp

    # --- Conversation and Tool Formatting ---

    def convert_conversation_history_to_adapter_format(
        self, conversation: Conversation, additional_parameters: Dict = None
    ) -> List[Dict]:
        """Converts a Conversation object into the format required by the Anthropic API."""
        if additional_parameters is None:
            additional_parameters = {}

        history = []
        citations_enabled = additional_parameters.get(
            "citations_enabled",
            additional_parameters.get("citations", False),
        )

        for message in conversation.messages:
            content = self._prepare_message_content(message, citations_enabled)
            history.append({"role": message.role, "content": content})

            # Per Anthropic's API, tool results must be in a separate, subsequent user message.
            if message.function_responses:
                tool_results_content = [fr.to_anthropic() for fr in message.function_responses]
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
                thinking_content = [tr.to_anthropic() for tr in message.thinking_responses]
                content_list = thinking_content + content_list  # Prepend thinking blocks

            if message.function_calls:
                tool_call_content = [fc.to_anthropic() for fc in message.function_calls]
                content_list.extend(tool_call_content)

        return content_list

    def _ensure_content_is_list(self, content: any) -> List[Dict]:
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
        if isinstance(file, PDFDocumentFile) and file.size < 32_000_000 and file.number_of_pages < 100:
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
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            param_type = PYTHON_TYPE_TO_JSON_SCHEMA.get(param.annotation, DEFAULT_JSON_SCHEMA_TYPE)
            properties[name] = {'type': param_type}
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return {
            'name': func.__name__,
            'description': func.__doc__ or '',
            'input_schema': {'type': 'object', 'properties': properties, 'required': required},
        }

    # --- Token Counting ---

    def count_tokens(self, model: str, messages: List[Dict], tools: List[Dict] = []) -> int:
        """Counts the number of input tokens for a given model, message list, and tools."""
        try:
            response = self.client.messages.count_tokens(model=model, messages=messages, tools=tools)
            return response.input_tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for model {model}: {e}")
            return 0

    def correct_max_tokens(self, model: str, messages: List[Dict], max_tokens: int, tools: List[Dict] = []) -> int:
        """Adjusts max_tokens to prevent exceeding the model's context window."""
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
    
    def request_llm_with_functions(self,
                                   model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool]=[], 
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs
                                   ): 
        """
        Not implemented
        This method is not implemented in the AnthropicAdapter.
        """
        raise NotImplementedError("request_llm_with_functions is not implemented in AnthropicAdapter.")
