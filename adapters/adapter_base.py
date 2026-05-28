import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from dotenv import load_dotenv
from loguru import logger

from llm_platform.services.conversation import Conversation, Message
from llm_platform.tools.base import BaseTool
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.types import AdditionalParameters


# Python type -> JSON-schema type mapping for converting plain callables to tool schemas
PYTHON_TYPE_TO_JSON_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
DEFAULT_JSON_SCHEMA_TYPE = "string"

# A PDF below BOTH thresholds is sent inline (base64); otherwise its extracted text is sent.
PDF_INLINE_MAX_BYTES = 32_000_000
PDF_INLINE_MAX_PAGES = 100

# Safety cap on agentic tool-calling rounds, to bound runaway cost / unbounded loops.
MAX_TOOL_ROUNDS = 20


class AdapterBase(ABC):
    """Base class for chat-LLM provider adapters.

    Subclasses convert a `Conversation` into the provider's wire format and
    implement `request_llm`. Provider-agnostic helpers shared by most adapters
    (parameter merging, usage extraction, callable->schema conversion, and
    content-block formatting) live here so the adapters stay thin.
    """

    def __init__(self):
        load_dotenv()
        self.latest_usage = None
        self.model_config = ModelConfig()

    # --- Abstract contract ---

    @abstractmethod
    def convert_conversation_history_to_adapter_format(
        self, the_conversation: Conversation, *args, **kwargs
    ):
        """Convert a Conversation into the provider-specific message list.

        Concrete adapters extend this signature with provider-specific extras
        (e.g. `model`, `additional_parameters`).
        """

    @abstractmethod
    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        """Send a single request to the provider and return the assistant Message."""

    async def request_llm_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        """Async entry point with a thread-offloaded default.

        Adapters backed by a native async SDK (e.g. OpenAI) override this; all
        other adapters inherit this default, which runs the synchronous
        ``request_llm`` off the event loop so the facade's async surface works
        for every provider instead of raising ``AttributeError``.
        """
        return await asyncio.to_thread(
            self.request_llm,
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
            **kwargs,
        )

    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ):
        """Tool-calling variant: resolve tool calls and re-ask until done.

        Adapters that support tool calling override this. The default makes the
        lack of support explicit and uniform instead of a per-adapter stub.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support tool calling")

    # --- Shared helpers ---

    def _merge_additional_parameters(
        self,
        additional_parameters: AdditionalParameters | None,
        kwargs: Dict,
    ) -> Dict:
        """Merge deprecated ``**kwargs`` into a copy of ``additional_parameters``."""
        merged_parameters = dict(additional_parameters or {})
        if kwargs:
            logger.warning(
                "Passing request parameters via **kwargs is deprecated; use additional_parameters."
            )
            for key, value in kwargs.items():
                merged_parameters.setdefault(key, value)
        return merged_parameters

    def _build_usage(
        self,
        usage,
        model: str,
        *,
        completion_attr: str = "completion_tokens",
        prompt_attr: str = "prompt_tokens",
    ) -> Dict:
        """Build the standard usage dict from a provider usage object.

        Missing values default to 0 so a missing/None usage object yields zeros
        instead of crashing (the provider attribute names vary, hence the
        configurable ``*_attr`` keyword arguments).
        """
        return {
            "model": model,
            "completion_tokens": getattr(usage, completion_attr, 0) if usage is not None else 0,
            "prompt_tokens": getattr(usage, prompt_attr, 0) if usage is not None else 0,
        }

    def _callable_to_json_schema(self, func: Callable) -> Dict:
        """Introspect a plain Python callable into a canonical JSON-schema tool definition.

        Returns ``{"name", "description", "parameters": {"type", "properties", "required"}}``;
        each adapter wraps this in its provider-specific envelope.
        """
        signature = inspect.signature(func)
        properties: Dict[str, Dict] = {}
        required: List[str] = []
        for name, parameter in signature.parameters.items():
            properties[name] = {
                "type": PYTHON_TYPE_TO_JSON_SCHEMA.get(parameter.annotation, DEFAULT_JSON_SCHEMA_TYPE)
            }
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {"type": "object", "properties": properties, "required": required},
        }

    def _image_data_url(self, file) -> str:
        """Return a base64 ``data:`` URL for an image file."""
        return f"data:image/{file.extension};base64,{file.base64}"

    def _document_xml(self, file) -> str:
        """Wrap a document's extracted text in a named ``<document>`` tag."""
        return f'<document name="{file.name}">{file.text}</document>'
