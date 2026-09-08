import asyncio
import functools
import inspect
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from dotenv import load_dotenv
from loguru import logger
from pydantic import TypeAdapter

from llm_platform.services.conversation import Conversation, Message
from llm_platform.tools.base import BaseTool
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.types import AdditionalParameters


# Load environment variables (.env) once when the adapters layer is first imported,
# rather than on every adapter construction.
load_dotenv()


# A PDF below BOTH thresholds is sent inline (base64); otherwise its extracted text is sent.
PDF_INLINE_MAX_BYTES = 32_000_000
PDF_INLINE_MAX_PAGES = 100

# Safety cap on agentic tool-calling rounds, to bound runaway cost / unbounded loops.
MAX_TOOL_ROUNDS = 40


class ResponseTimeoutError(TimeoutError):
    """A polling/continuation deadline expired; the last response remains available."""

    def __init__(self, response=None):
        self.response = response
        self.response_id = getattr(response, "id", None)
        super().__init__(f"Response deadline exceeded (response_id={self.response_id})")


class AdapterBase(ABC):
    """Base class for chat-LLM provider adapters.

    Subclasses convert a `Conversation` into the provider's wire format and
    implement `request_llm`. Provider-agnostic helpers shared by most adapters
    (parameter merging, usage extraction, callable->schema conversion, and
    content-block formatting) live here so the adapters stay thin.
    """

    RESPONSE_TIMEOUT_SECONDS = 1800

    @staticmethod
    def _check_deadline(deadline, response=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResponseTimeoutError(response)
        return remaining

    def __init__(self):
        self.latest_usage = None
        self.model_config = ModelConfig()
        self._client = None

    @property
    def client(self):
        """Provider SDK client, constructed lazily once on first access."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        """Construct the provider SDK client. Subclasses override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build_client()")

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

    @staticmethod
    def _tool_name(func) -> str:
        """Declared tool name for a ``BaseTool`` instance or a plain callable."""
        if isinstance(func, BaseTool):
            return func.name
        if isinstance(func, functools.partial):
            return AdapterBase._tool_name(func.func)
        return getattr(func, "__name__", type(func).__name__)

    def _callable_to_json_schema(self, func: Callable) -> Dict:
        """Introspect a plain Python callable into a canonical JSON-schema tool definition.

        Pydantic resolves annotations and retains defaults, unions, containers,
        constraints, and recursive definitions. Unannotated inputs accept any
        JSON value. Tool arguments are passed as keyword arguments, so signatures
        that cannot be represented by a fixed keyword object are rejected.
        """
        signature = inspect.signature(func)
        for name, parameter in signature.parameters.items():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError(
                    f"Tool {self._tool_name(func)!r} parameter {name!r} must be "
                    "a named keyword-compatible parameter; positional-only and "
                    "variadic parameters are unsupported."
                )
        # Pydantic accepts functions, bound methods, and partials directly;
        # callable instances expose their annotations on the bound __call__.
        target = func
        if not inspect.isroutine(func) and not isinstance(func, functools.partial):
            target = func.__call__
        try:
            parameters = TypeAdapter(target).json_schema()
        except (TypeError, ValueError, NameError) as exc:
            raise TypeError(
                f"Cannot generate JSON Schema for tool {self._tool_name(func)!r}: {exc}"
            ) from exc
        return {
            "name": self._tool_name(func),
            "description": inspect.getdoc(func) or "",
            "parameters": BaseTool.clean_schema(parameters),
        }

    def _image_data_url(self, file) -> str:
        """Return a base64 ``data:`` URL for an image file."""
        return f"data:image/{file.extension};base64,{file.base64}"

    def _document_xml(self, file) -> str:
        """Wrap a document's extracted text in a named ``<document>`` tag."""
        return f'<document name="{file.name}">{file.text}</document>'
