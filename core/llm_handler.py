from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.openai_image_adapter import OpenAIImageAdapter
from llm_platform.adapters.openai_old_adapter import OpenAIOldAdapter
from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.adapters.speechmatics_adapter import SpeechmaticsAdapter
from llm_platform.adapters.assemblyai_adapter import AssemblyAIAdapter
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.grok_adapter import GrokAdapter
from llm_platform.adapters.grok_image_adapter import GrokImageAdapter
from llm_platform.adapters.deepseek_adapter import DeepSeekAdapter
from llm_platform.adapters.elevenlabs_adapter import ElevenLabsAdapter
from llm_platform.adapters.mistral_adapter import MistralAdapter
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
from llm_platform.tools.base import BaseTool
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.types import AdditionalParameters
from loguru import logger
import tiktoken
import os
from typing import Any, BinaryIO, Callable, Dict, List, Tuple, Union

class APIHandler:
    """
    APIHandler is a class that handles interactions with various language model adapters. 
    It provides methods for making synchronous and asynchronous requests to language models, converting voice to text, 
    generating images, and retrieving available models.
    
    Attributes:
        adapters (Dict[str, Any]): A dictionary to store initialized adapters.
        model_config (ModelConfig): Configuration for the language models.
        the_conversation (Conversation): The conversation context.
        history (List): Legacy attribute for storing history.
        current_costs (Any): Legacy attribute for storing current costs.
    Methods:
        __init__(system_prompt: str = "You are a helpful assistant"):
            Initializes the APIHandler with a system prompt.
        _lazy_initialization_of_adapter(adapter_name: str) -> Any:
            Lazily initializes and returns the specified adapter.
        get_adapter(model_name: str) -> Any:
            Gets the appropriate adapter for the given model name.
        request(model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None, files: List[BaseFile] = [], tool_output_callback: Callable = None, additional_parameters: AdditionalParameters | None = None, **kwargs) -> str:
        request_async(model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None, files: List[BaseFile] = [], tool_output_callback: Callable = None, additional_parameters: AdditionalParameters | None = None, **kwargs) -> str:
            Asynchronously sends a request to the language model with the given parameters.
        request_llm(model: str, functions: Union[List[BaseTool], List[Callable]] = None, temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: AdditionalParameters | None = None, **kwargs) -> str:
            Makes a request to the language model using the appropriate adapter.
        request_llm_async(model: str, functions: Union[List[BaseTool], List[Callable]] = None, temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: AdditionalParameters | None = None, **kwargs) -> str:
            Asynchronously makes a request to the language model using the appropriate adapter.
        calculate_tokens(text: str) -> Dict[str, int]:
            Calculates the number of tokens in the given text.
        voice_to_text(audio_file: BinaryIO, audio_format: str, provider: str = 'openai', **kwargs) -> str:
            Converts voice to text using the specified provider.
        voice_file_to_text(audio_file_name: str, provider: str = 'openai', **kwargs) -> str:
            Converts a voice file to text using the specified provider.
        generate_image(prompt: str, provider: str = 'openai', n: int = 1, **kwargs) -> Union[str, List[str]]:
        get_models(adapter_name: str) -> List[str]:
            Retrieves the available models for the specified adapter.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the APIHandler.

        Args:
            system_prompt (str): The system prompt to use for the conversation.
        """
        self.adapters: Dict[str, Any] = {}
        self.model_config = ModelConfig()
        self.the_conversation = Conversation(system_prompt = system_prompt)

        # Legacy:
        self.history = []
        self.current_costs = None

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
            adapter_class = {
                "OpenAIAdapter": OpenAIAdapter,
                "OpenAIImageAdapter": OpenAIImageAdapter,
                "AnthropicAdapter": AnthropicAdapter,
                "OpenRouterAdapter": OpenRouterAdapter,
                "SpeechmaticsAdapter": SpeechmaticsAdapter,
                "ElevenLabsAdapter": ElevenLabsAdapter,
                "AssemblyAIAdapter": AssemblyAIAdapter,
                "GoogleAdapter": GoogleAdapter,
                "GrokAdapter": GrokAdapter,
                "GrokImageAdapter": GrokImageAdapter,
                "DeepSeekAdapter": DeepSeekAdapter,
                "MistralAdapter": MistralAdapter,
                "OpenAIOldAdapter": OpenAIOldAdapter,
            }.get(adapter_name)

            if adapter_class is None:
                raise ValueError(f"Adapter {adapter_name} is not supported")

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
        adapter_name = self.model_config[model_name].adapter
        #print(f"Using adapter: {adapter_name} for model: {model_name}")
        return self._lazy_initialization_of_adapter(adapter_name)

    @staticmethod
    def _set_nested_parameter(target: Dict[str, Any], path: str, value: Any) -> None:
        if not path:
            return
        parts = [part for part in str(path).split(".") if part]
        if not parts:
            return
        cursor = target
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    def _allowed_parameter_keys(self, model_object: Any) -> set:
        if not model_object:
            return set()
        allowed = set()
        for name, definition in model_object.parameter_definitions().items():
            if definition.get("include_in_request", True):
                allowed.add(name)
            request_key = definition.get("request_key")
            if request_key:
                allowed.add(str(request_key).split(".")[0])
        allowed.update({"response_modalities"})
        return allowed

    def _prepare_additional_parameters(
        self,
        model: str,
        additional_parameters: AdditionalParameters | None,
        **kwargs,
    ) -> Dict:
        """
            - It takes whatever the caller passed in additional_parameters and merges in any deprecated **kwargs (with a warning).
            - It looks up the model in models_config.yaml and applies defaults for that model’s extra parameters (only when send_default is true), including max_tokens.
            - It remaps “friendly” parameter names into nested provider formats using request_key (e.g. reasoning_effort → reasoning.effort, max_tokens → max_output_tokens), and drops empty values while doing so.
            - It removes parameters marked include_in_request: false.
            - It filters out anything the model doesn’t support, logging a warning for each unsupported key.

            Concrete example
                If a model defines:

                reasoning_effort:
                    type: enum
                    default: none
                    request_key: reasoning.effort

                and you pass:

                additional_parameters={"reasoning_effort": "high"}

                _prepare_additional_parameters will turn that into:

                {"reasoning": {"effort": "high"}}

                and only send it if the model allows it.
        """
        merged: Dict[str, Any] = {}
        if additional_parameters:
            merged.update(additional_parameters)

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                merged.setdefault(key, value)

        model_object = self.model_config[model] if model else None
        if model_object:
            for name, definition in model_object.parameter_definitions().items():
                if not definition.get("send_default", True):
                    continue
                if name in merged:
                    continue
                request_key = definition.get("request_key")
                if request_key:
                    root_key = str(request_key).split(".")[0]
                    if root_key in merged:
                        continue
                if "default" in definition and definition["default"] is not None:
                    merged[name] = definition["default"]

            for name, definition in model_object.parameter_definitions().items():
                request_key = definition.get("request_key")
                if request_key and name in merged:
                    value = merged.pop(name)
                    if value is None or value == "":
                        continue
                    self._set_nested_parameter(merged, request_key, value)

            for name, definition in model_object.parameter_definitions().items():
                if not definition.get("include_in_request", True):
                    merged.pop(name, None)

            allowed = self._allowed_parameter_keys(model_object)
            if allowed:
                filtered: Dict[str, Any] = {}
                for key, value in merged.items():
                    if key in allowed:
                        filtered[key] = value
                    else:
                        logger.warning(
                            f"Model {model} does not support parameter '{key}'. Ignoring the parameter."
                        )
                merged = filtered

        return merged

    def request(self, 
                model: str, 
                prompt: str, 
                functions: Union[List[BaseTool], List[Callable]] = None, 
                files: List[BaseFile]=[],
                tool_output_callback: Callable=None,
                additional_parameters: AdditionalParameters | None = None,
                **kwargs) -> str:
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
        # Add prompt to the conversation
        message = Message(role="user", 
                          content=prompt, 
                          usage=None, 
                          files=files)
        self.the_conversation.messages.append(message)

        normalized_parameters = self._prepare_additional_parameters(
            model,
            additional_parameters,
            **kwargs,
        )

        return self.request_llm(
            model,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=normalized_parameters,
        )
        
    async def request_async(self, 
                model: str, 
                prompt: str, 
                functions: Union[List[BaseTool], List[Callable]] = None, 
                files: List[BaseFile]=[],
                tool_output_callback: Callable=None,
                additional_parameters: AdditionalParameters | None = None,
                **kwargs) -> str:
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
        
        # Add prompt to the conversation
        message = Message(role="user", 
                          content=prompt, 
                          usage=None, 
                          files=files)
        self.the_conversation.messages.append(message)

        normalized_parameters = self._prepare_additional_parameters(
            model,
            additional_parameters,
            **kwargs,
        )

        return await self.request_llm_async(
            model,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=normalized_parameters,
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

        try:
            response = adapter.request_llm(
                model=model,
                the_conversation=self.the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=normalized_parameters,
            )

            return response
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            message = Message(role="assistant", content=f"Error in API call: {e}", usage=None, files=[])
            self.the_conversation.messages.append(message)
            return message
    
    async def request_llm_async(self,
                        model: str,
                        functions: Union[List[BaseTool], List[Callable]] = None,
                        tool_output_callback: Callable=None,
                        additional_parameters: AdditionalParameters | None = None,
                        **kwargs) -> str:
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
    def calculate_tokens(text) -> Dict[str, int]:
        """
        Calculate the number of tokens in the given text.

        Args:
            text (str): The text to calculate tokens for.

        Returns:
            Dict[str, int]: A dictionary containing the byte count and token count.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return {'bytes': len(text),
                  'tokens': num_tokens}
    
    def voice_to_text(self, audio_file: BinaryIO, audio_format: str, provider: str='openai', **kwargs):
        assert provider in ['openai', 'speechmatics', 'elevenlabs'], f"Provider {provider} is not supported. I understand only 'openai' or 'speechmatics' or 'elevenlabs' as providers."
        
        if provider.lower() == 'openai':
            adapter =  self._lazy_initialization_of_adapter('OpenAIAdapter')
            response_format = kwargs.get('response_format', 'text') 
            language = kwargs.get('language', 'en')
            model = kwargs.get('model', 'whisper-1')
            transcript = adapter.voice_to_text(audio_file, response_format, language, model=model)
            return transcript
        
        elif provider.lower() == 'speechmatics':
            adapter =  self._lazy_initialization_of_adapter('SpeechmaticsAdapter')
            language = kwargs.get('language', 'en')
            transcription_config = kwargs.get('transcription_config', None)
            transcript = adapter.voice_to_text(('audio_file.'+audio_format, audio_file), language, transcription_config)
            return transcript
        
        elif provider.lower() == 'elevenlabs':
            adapter =  self._lazy_initialization_of_adapter('ElevenLabsAdapter')
            language = kwargs.get('language', 'eng')
            diarized = kwargs.get('diarized', True)
            transcript = adapter.voice_to_text(audio_file, audio_format, language, diarized)
            return transcript
        
        else: 
            raise ValueError(f"Provider {provider} is not supported. I understand only 'openai' or 'speechmatics' as providers. ")

    def voice_file_to_text(self, audio_file_name: str, provider: str='openai', **kwargs):
        # Get file type
        filename, file_extension = os.path.splitext(audio_file_name)
        file_extension = file_extension.lower().replace(".", "")

        with open(audio_file_name, 'rb') as audio_file:
            transcript = self.voice_to_text(audio_file, file_extension, provider, **kwargs)
            return transcript

    def generate_image(self, prompt: str, provider: str='openai', n=1, **kwargs):
        """
        Generates an image based on the given prompt using the specified provider.

        Args:
            prompt (str): A textual description of the desired image content.
            provider (str, optional): The image generation provider to use. 
                Supported values are 'openai' and 'google'. Defaults to 'openai'.
            n (int, optional): The number of images to generate. Defaults to 1.
            **kwargs: Additional parameters specific to the chosen provider.

        Keyword Arguments:
            For the OpenAI adapter:
                - size (str, optional): Supported values are '1024x1024', '1024x1536', '1536x1024', 'auto'.
                - quality (str, optional): Supported values are 'low', 'medium', 'high', 'auto'.
                - moderation (str, optional): Supported values are 'low', 'auto'

            For the Google adapter:
                - negative_prompt (str, optional): A description of what to omit in the generated image.
                - number_of_images (int, optional): The number of images to generate, from 1 to 4. Defaults to 4.
                - aspect_ratio (str, optional): Supported values are '1:1', '3:4', '4:3', '9:16', and '16:9'. Defaults to '1:1'.
                - safety_filter_level (str, optional): Supported values are:
                    - 'block_low_and_above'
                    - 'block_medium_and_above'
                    - 'block_only_high'
                - person_generation (str, optional): Supported values are:
                    - 'dont_allow': Block generation of images of people.
                    - 'allow_adult': Allow generation of adult images but block children.
                - quality (str, optional): Supported values are 'low', 'medium', 'high', 'auto'.
                    Converts by adapter into proper `model` and `image_size`.

        Returns:
            list or str: The generated image(s) URL(s) or list of images, depending on the provider.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        if provider.lower() == 'openai':
            adapter = self._lazy_initialization_of_adapter("OpenAIAdapter")
            image_url = adapter.generate_image(prompt, n, **kwargs)
            return image_url
        elif provider.lower() == 'google':
            adapter = self._lazy_initialization_of_adapter("GoogleAdapter")
            images = adapter.generate_image(prompt, n, **kwargs)
            return images
        elif provider.lower() == 'grok':
            adapter = self._lazy_initialization_of_adapter("GrokAdapter")
            images = adapter.generate_image(prompt, n, **kwargs) # returns List[ImageFile]
            return images
        else: 
            raise ValueError(f"Provider {provider} is not supported. I understand only 'openai' or 'google' as providers. ")
        
    async def generate_video(self, prompt: str, provider: str='google', **kwargs):
        """
        Generates an video based on the given prompt using the specified provider.

        Args:
            prompt (str): A textual description of the desired image content.
            provider (str, optional): The image generation provider to use. 
                Supported value is 'google'.
            n (int, optional): The number of videos to generate. Defaults to 1.
            **kwargs: Additional parameters specific to the chosen provider.

        Keyword Arguments:
            For the Google adapter:
                - negative_prompt (str, optional): A description of what to omit in the generated video.

        Returns:
            list or str: The generated image(s) URL(s) or list of images, depending on the provider.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        if provider.lower() == 'google':
            adapter = self._lazy_initialization_of_adapter("GoogleAdapter")
            videos = adapter.generate_video(prompt, **kwargs)
            return videos
        elif provider.lower() == 'openai':
            adapter = self._lazy_initialization_of_adapter("OpenAIAdapter")
            videos = await adapter.generate_video(prompt, **kwargs)
            return videos
        else: 
            raise ValueError(f"Provider {provider} is not supported. I understand only 'google' as provider.")
        
    def edit_image(self, prompt: str, provider: str='openai', images=List[ImageFile], n=1, **kwargs):
        """
        Edits an image based on the given prompt using the specified provider. 
        Now works only with OpenAI.

        Args:
            prompt (str): A textual description of the desired image content.
            provider (str, optional): The image generation provider to use. 
                Supported values is 'openai'.
            n (int, optional): The number of images to generate. Defaults to 1.
            **kwargs: Additional parameters specific to the chosen provider.

        Keyword Arguments:
            For the OpenAI adapter:
                - size (str, optional): Supported values are '1024x1024', '1024x1536', '1536x1024', 'auto'.
                - quality (str, optional): Supported values are 'low', 'medium', 'high', 'auto'.
                - moderation (str, optional): Supported values are 'low', 'auto'
                - input_fidelity (str, optional): Supported values are 'low', 'high'. The default value is `low`.
                     Input fidelity, which allows you to better preserve details from the input images in the output.
        """
        if provider.lower() == 'openai':
            adapter = self._lazy_initialization_of_adapter("OpenAIAdapter")
            image_url = adapter.edit_image(prompt, images, n, **kwargs)
            return image_url
        else: 
            raise ValueError(f"Provider {provider} is not supported. I understand only 'openai' as providers. ")

    def get_models(self, adapter_name: str) -> List[str]:

        adapter = self._lazy_initialization_of_adapter(adapter_name)

        models = adapter.get_models()
        return models
