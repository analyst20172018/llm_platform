from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Callable, List
from llm_platform.services.conversation import Conversation, Message
from llm_platform.tools.base import BaseTool
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.types import AdditionalParameters


class AdapterBase(ABC):
    """Base class for chat-LLM provider adapters.

    Subclasses convert a `Conversation` into the provider's wire format and
    expose `request_llm` plus `request_llm_with_functions` for tool-calling.
    Speech-only adapters do not inherit from this class.
    """

    def __init__(self):
        load_dotenv()
        self.latest_usage = None
        self.model_config = ModelConfig()

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

    @abstractmethod
    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ):
        """Tool-calling variant: resolve tool calls and re-ask until done."""
