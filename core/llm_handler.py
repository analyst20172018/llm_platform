import importlib
from typing import Any, Callable, Dict, List, Union

import tiktoken
from loguru import logger

from llm_platform.core.parameter_normalizer import ParameterNormalizer
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import BaseFile, ImageFile, PDFDocumentFile
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters

# Adapter classes are imported lazily (on first use) so importing the facade does
# not transitively import every provider SDK. Maps adapter name -> "module:ClassName".
ADAPTER_IMPORT_PATHS = {
    "OpenAIAdapter": "llm_platform.adapters.openai_adapter:OpenAIAdapter",
    "AnthropicAdapter": "llm_platform.adapters.anthropic_adapter:AnthropicAdapter",
    "OpenRouterAdapter": "llm_platform.adapters.openrouter_adapter:OpenRouterAdapter",
    "GoogleAdapter": "llm_platform.adapters.google_adapter:GoogleAdapter",
    "GrokAdapter": "llm_platform.adapters.grok_adapter:GrokAdapter",
    "DeepSeekAdapter": "llm_platform.adapters.deepseek_adapter:DeepSeekAdapter",
    "MistralAdapter": "llm_platform.adapters.mistral_adapter:MistralAdapter",
}

class APIHandler:
    """
    APIHandler is a class that handles interactions with various language model adapters.
    It provides methods for making synchronous and asynchronous requests to language models
    and retrieving available models.
    
    Adapter selection, parameter normalization, and conversation state are all
    handled here; provider-specific translation lives in the adapters.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the APIHandler.

        Args:
            system_prompt (str): The system prompt to use for the conversation.
        """
        self.adapters: Dict[str, Any] = {}
        self.model_config = ModelConfig()
        self.normalizer = ParameterNormalizer(self.model_config)
        self.the_conversation = Conversation(system_prompt = system_prompt)

    def _lazy_initialization_of_adapter(self, adapter_name: str):
        """
        Lazily initialize and return the specified adapter.

        Args:
            adapter_name (str): The name of the adapter to initialize.

        Returns:
            Any: The initialized adapter.

        Raises:
            ValueError: If the specified adapter is not supported.
        """
        if adapter_name not in self.adapters:
            import_path = ADAPTER_IMPORT_PATHS.get(adapter_name)
            if import_path is None:
                raise ValueError(f"Adapter {adapter_name} is not supported")

            module_path, class_name = import_path.split(":")
            adapter_class = getattr(importlib.import_module(module_path), class_name)
            self.adapters[adapter_name] = adapter_class()

        return self.adapters[adapter_name]
    
    def get_adapter(self, model_name: str) -> Any:
        """
        Get the appropriate adapter for the given model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            Any: The adapter for the specified model.
        """
        model_object = self.model_config[model_name]
        if model_object is None:
            raise ValueError(f"Model '{model_name}' is not defined in models_config.yaml")
        return self._lazy_initialization_of_adapter(model_object.adapter)

    def _append_user_message(self, prompt: str, files: List[BaseFile] | None) -> None:
        self.the_conversation.messages.append(
            Message(
                role="user",
                content=prompt,
                usage=None,
                files=[] if files is None else files,
            )
        )

    def _prepare_additional_parameters(
        self,
        model: str,
        additional_parameters: AdditionalParameters | None,
        **kwargs,
    ) -> Dict:
        """Normalize request parameters against the selected model definition."""
        return self.normalizer.normalize(model, additional_parameters, **kwargs)

    def request(
        self,
        model: str,
        prompt: str,
        functions: Union[List[BaseTool], List[Callable]] = None,
        files: List[BaseFile] | None = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        """
            Send a single prompt to a language model and receive the assistant’s answer.

            The method

            1.  Appends the ``prompt`` (and optional ``files``) to the current
                ``Conversation`` as a *user* message.
            2.  Forwards the full conversation to the appropriate *adapter* that
                serves the requested ``model``.
            3.  Returns the assistant’s reply as a ``Message`` instance and stores
                it in the conversation history.

            Parameters
            ----------
            model : str
                Name of the model to use.  
                Must correspond to a model present in *models_config.yaml*.
            prompt : str
                Natural‑language question / instruction supplied by the user.
            functions : list[BaseTool | Callable], optional
                Collection of tools that the model may call via function‑calling
                interfaces (OpenAI, Gemini, Anthropic, …).  Each element must be

                * an instance of ``llm_platform.tools.base.BaseTool`` **or**
                * a plain Python function whose signature will be converted to a
                JSON schema.
            files : list[BaseFile], optional
                Additional multimodal inputs (images, audio, PDFs, Excel sheets,
                plain text files, …).  
                See ``llm_platform.services.files`` for available file classes.
            tool_output_callback : Callable, optional
                ``callback(tool_name: str, args: list, result: Any) -> None``  
                Invoked after every successful tool execution.
            additional_parameters : AdditionalParameters, optional
                Provider‑agnostic high‑level switches.  Currently understood keys (silently ignored by adapters that do not support them):
                  How additional_parameters works in the program overall
                    - You pass additional_parameters to APIHandler.request(...) or APIHandler.request_llm(...).
                    - The handler normalizes and validates them via _prepare_additional_parameters.
                    - The cleaned dict is then passed to the adapter for the selected provider (OpenAI, Anthropic, Google, etc.).
                    - Each adapter picks the keys it understands and translates them into provider‑specific request fields. Unsupported keys are ignored (or already filtered out).
                    - The list of “allowed” parameters and defaults is defined per model in models_config.yaml under additional_parameters. This includes UI hints, defaults, and request_key mappings for nested fields.
                * Every possible additional parameter is described in the file `llm_platform\types.py` under the `AdditionalParameters` type alias.
            **kwargs
                Deprecated: keyword arguments are merged into ``additional_parameters`` for backwards compatibility.

            Returns
            -------
            llm_platform.services.conversation.Message
                The assistant’s reply, including text, any returned files,
                reasoning traces, function‑call metadata and token usage.

            Raises
            ------
            ValueError
                If *prompt* is empty.

            Examples
            --------
            >>> handler = APIHandler()
            >>> reply = handler.request(
            ...     model="gpt-4o",
            ...     prompt="Summarise the attached PDF in bullet points.",
            ...     files=[PDFDocumentFile.from_bytes(pdf_bytes, "report.pdf")],
            ...     additional_parameters={"web_search": True},
            ... )
            >>> print(reply.content)
            • …

            Notes
            -----
            All messages exchanged through this method are persisted in
            ``handler.the_conversation``.  Subsequent calls therefore provide
            conversational context to the model until ``Conversation.clear()`` is
            invoked.
            """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        self._append_user_message(prompt, files)

        return self.request_llm(
            model,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
            **kwargs,
        )

    async def request_async(
        self,
        model: str,
        prompt: str,
        functions: Union[List[BaseTool], List[Callable]] = None,
        files: List[BaseFile] | None = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        """
        Make a request to the language model.

        Args:
            model (str): The name of the model to use.
            prompt (str): The prompt to send to the model.
            images (Optional[List[ImageFile]]): A list of image files to include in the request.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            str: The response from the language model.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        self._append_user_message(prompt, files)

        return await self.request_llm_async(
            model,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
            **kwargs,
        )
    
    def request_llm(self, 
                    model: str, 
                    functions: Union[List[BaseTool], List[Callable]] = None, 
                    tool_output_callback: Callable=None,
                    additional_parameters: AdditionalParameters | None = None, 
                    **kwargs) -> Message:
        """
            Dispatch the current ``Conversation`` to the language‑model adapter and
            return the assistant’s next message.

            Unlike :py:meth:`APIHandler.request`, this method does **not** append a
            new *user* message.  It is therefore intended to be called internally
            (e.g. by :py:meth:`request`) or when the caller has already modified
            ``self.the_conversation`` manually.

            Workflow
            --------
            1.  Select the adapter that corresponds to ``model`` based on
                *models_config.yaml*.
            2.  Normalize ``additional_parameters`` against the model's YAML schema
                (apply defaults, map ``request_key`` fields, filter unsupported keys).
            3.  Forward the entire conversation as well as all arguments to
                ``adapter.request_llm``.
            4.  Store the returned assistant message inside the conversation and
                return it to the caller.

            Parameters
            ----------
            model : str
                The identifier of the model to invoke.
            functions : list[BaseTool | Callable], optional
                Tools that the model is allowed to call.  See
                :py:meth:`APIHandler.request` for details.
            tool_output_callback : Callable, optional
                ``callback(tool_name: str, args: list, result: Any)`` executed after
                every tool call.
            additional_parameters : AdditionalParameters, optional
                Provider‑agnostic high‑level switches.  Currently understood keys (silently ignored by adapters that do not support them):
                  How additional_parameters works in the program overall
                    - You pass additional_parameters to APIHandler.request(...) or APIHandler.request_llm(...).
                    - The handler normalizes and validates them via _prepare_additional_parameters.
                    - The cleaned dict is then passed to the adapter for the selected provider (OpenAI, Anthropic, Google, etc.).
                    - Each adapter picks the keys it understands and translates them into provider‑specific request fields. Unsupported keys are ignored (or already filtered out).
                    - The list of “allowed” parameters and defaults is defined per model in models_config.yaml under additional_parameters. This includes UI hints, defaults, and request_key mappings for nested fields.
                * Every possible additional parameter is described in the file `llm_platform\types.py` under the `AdditionalParameters` type alias.
            **kwargs
                Deprecated: keyword arguments are merged into ``additional_parameters`` for backwards compatibility.

            Returns
            -------
            llm_platform.services.conversation.Message
                The newly generated assistant message, complete with any attached
                files, internal “thinking” traces, tool‑use records and token usage
                statistics.

            Raises
            ------
            ValueError
                If *model* does not exist in *models_config.yaml* or the associated
                adapter cannot be instantiated.

            Notes
            -----
            * The conversation history grows with every invocation.  Call
            ``handler.the_conversation.clear()`` to reset the context.
        """
        logger.info(f"Calling API with model {model}")
        adapter = self.get_adapter(model)

        normalized_parameters = self._prepare_additional_parameters(
            model,
            additional_parameters,
            **kwargs,
        )

        return adapter.request_llm(
            model=model,
            the_conversation=self.the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=normalized_parameters,
        )
    
    async def request_llm_async(self,
                        model: str,
                        functions: Union[List[BaseTool], List[Callable]] = None,
                        tool_output_callback: Callable=None,
                        additional_parameters: AdditionalParameters | None = None,
                        **kwargs) -> Message:
        logger.info(f"Calling API asynchronously with model {model}")
        adapter = self.get_adapter(model)

        normalized_parameters = self._prepare_additional_parameters(
            model,
            additional_parameters,
            **kwargs,
        )

        response = await adapter.request_llm_async(
            model=model,
            the_conversation=self.the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=normalized_parameters,
        )
        return response

    @staticmethod
    def calculate_tokens(text: str) -> Dict[str, int]:
        """
        Calculate the number of tokens in the given text.

        Args:
            text (str): The text to calculate tokens for.

        Returns:
            Dict[str, int]: A dictionary containing the byte count and token count.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return {"bytes": len(text), "tokens": num_tokens}
