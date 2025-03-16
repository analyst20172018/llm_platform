from typing import List, Dict
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.openai_old_adapter import OpenAIOldAdapter
from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.adapters.openrouter_adapter import OpenRouterAdapter
from llm_platform.adapters.speechmatics_adapter import SpeechmaticsAdapter
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.grok_adapter import GrokAdapter
from llm_platform.adapters.deepseek_adapter import DeepSeekAdapter
from llm_platform.adapters.elevenlabs_adapter import ElenenlabsAdapter
from llm_platform.adapters.mistral_adapter import MistralAdapter
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
from llm_platform.tools.base import BaseTool
from llm_platform.helpers.model_config import ModelConfig
import logging
import tiktoken
import os
import base64
from typing import List, Dict, Tuple, BinaryIO, Any, Callable, Union
import yaml

class APIHandler:
    """
    APIHandler is a class that handles interactions with various language model adapters. 
    It provides methods for making synchronous and asynchronous requests to language models, converting voice to text, 
    generating images, and retrieving available models.
    
    Attributes:
        adapters (Dict[str, Any]): A dictionary to store initialized adapters.
        model_config (ModelConfig): Configuration for the language models.
        the_conversation (Conversation): The conversation context.
        logging_level (int): The logging level for the handler.
        history (List): Legacy attribute for storing history.
        current_costs (Any): Legacy attribute for storing current costs.
    Methods:
        __init__(system_prompt: str = "You are a helpful assistant", logging_level=logging.INFO):
            Initializes the APIHandler with a system prompt and logging level.
        _lazy_initialization_of_adapter(adapter_name: str) -> Any:
            Lazily initializes and returns the specified adapter.
        get_adapter(model_name: str) -> Any:
            Gets the appropriate adapter for the given model name.
        request(model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None, files: List[BaseFile] = [], temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: Dict = {}, **kwargs) -> str:
        request_async(model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None, files: List[BaseFile] = [], temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: Dict = {}, **kwargs) -> str:
            Asynchronously sends a request to the language model with the given parameters.
        request_llm(model: str, functions: Union[List[BaseTool], List[Callable]] = None, temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: Dict = {}, **kwargs) -> str:
            Makes a request to the language model using the appropriate adapter.
        request_llm_async(model: str, functions: Union[List[BaseTool], List[Callable]] = None, temperature: int = 0, tool_output_callback: Callable = None, additional_parameters: Dict = {}, **kwargs) -> str:
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
    
    def __init__(self, system_prompt: str = "You are a helpful assistant", logging_level=logging.INFO):
        """
        Initialize the APIHandler.

        Args:
            system_prompt (str): The system prompt to use for the conversation.
        """
        self.adapters: Dict[str, Any] = {}
        self.model_config = ModelConfig()
        self.the_conversation = Conversation(system_prompt = system_prompt)

        # Configure logging
        self.logging_level = logging_level
        logging.basicConfig(level=self.logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

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
                "AnthropicAdapter": AnthropicAdapter,
                "OpenRouterAdapter": OpenRouterAdapter,
                "SpeechmaticsAdapter": SpeechmaticsAdapter,
                "ElenenlabsAdapter": ElenenlabsAdapter,
                "GoogleAdapter": GoogleAdapter,
                "GrokAdapter": GrokAdapter,
                "DeepSeekAdapter": DeepSeekAdapter,
                "MistralAdapter": MistralAdapter,
                "OpenAIOldAdapter": OpenAIOldAdapter,
            }.get(adapter_name)

            if adapter_class is None:
                raise ValueError(f"Adapter {adapter_name} is not supported")

            self.adapters[adapter_name] = adapter_class(logging_level=self.logging_level)

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
        return self._lazy_initialization_of_adapter(adapter_name)

    def request(self, 
                model: str, 
                prompt: str, 
                functions: Union[List[BaseTool], List[Callable]] = None, 
                files: List[BaseFile]=[],
                temperature: int=0, 
                tool_output_callback: Callable=None,
                additional_parameters: Dict={},
                **kwargs) -> str:
        """
            Sends a request to the language model with the given parameters.
            Args:
                model (str): The name of the model to use.
                prompt (str): The prompt to send to the model.
                functions (Union[List[BaseTool], List[Callable]], optional): A list of tools or callables to use. Defaults to None.
                files (List[BaseFile], optional): A list of files to include in the request. Defaults to an empty list.
                temperature (int, optional): The temperature setting for the model. Defaults to 0.
                tool_output_callback (Callable, optional): A callback function for tool output. Defaults to None.
                additional_parameters (Dict, optional): Additional parameters to include in the request. Defaults to an empty dictionary.
                **kwargs: Additional keyword arguments.
            Returns:
                str: The response from the language model.
            Raises:
                ValueError: If the prompt is empty.
            """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Add prompt to the conversation
        message = Message(role="user", 
                          content=prompt, 
                          usage=None, 
                          files=files)
        self.the_conversation.messages.append(message)

        return self.request_llm(model,
                                functions=functions,
                                temperature=temperature,
                                tool_output_callback=tool_output_callback,
                                additional_parameters=additional_parameters,
                                **kwargs)
        
    async def request_async(self, 
                model: str, 
                prompt: str, 
                functions: Union[List[BaseTool], List[Callable]] = None, 
                files: List[BaseFile]=[],
                temperature: int=0, 
                tool_output_callback: Callable=None,
                additional_parameters: Dict={},
                **kwargs) -> str:
        """
        Make a request to the language model.

        Args:
            model (str): The name of the model to use.
            prompt (str): The prompt to send to the model.
            temperature (float): The temperature for text generation.
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

        return await self.request_llm_async(model,
                                functions=functions,
                                temperature=temperature,
                                tool_output_callback=tool_output_callback,
                                additional_parameters=additional_parameters,
                                **kwargs)
    
    def request_llm(self, 
                    model: str, 
                    functions: Union[List[BaseTool], List[Callable]] = None, 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={}, 
                    **kwargs) -> str:
        """
        Make a request to the language model using the appropriate adapter.

        Args:
            model (str): The name of the model to use.
            temperature (float): The temperature for text generation.
            **kwargs: Additional keyword arguments for the request.
            additional_parameters (Dict, optional): Additional parameters to include in the request:
                response_modalities=['text', 'image', 'audio']: The response modalities to include in the response (for Gemini and OpenAI)

        Returns:
            str: The response from the language model.
        """
        logging.info(f"Calling API with model {model}")
        adapter = self.get_adapter(model)

        # Fetch max_tokens from the model config
        if not "max_tokens" in kwargs:
            kwargs["max_tokens"] = self.model_config[model].max_tokens

        response = adapter.request_llm(model=model, 
                                       the_conversation=self.the_conversation, 
                                       functions=functions,
                                       temperature = temperature,
                                       tool_output_callback=tool_output_callback,
                                       additional_parameters=additional_parameters,
                                       **kwargs)

        return response
    
    async def request_llm_async(self,
                        model: str,
                        functions: Union[List[BaseTool], List[Callable]] = None,
                        temperature: int=0,
                        tool_output_callback: Callable=None,
                        additional_parameters: Dict={},
                        **kwargs) -> str:
        logging.info(f"Calling API asynchronously with model {model}")
        adapter = self.get_adapter(model)

        # Fetch max_tokens from the model config
        max_tokens = self.model_config[model].max_tokens

        response = await adapter.request_llm_async(
            model=model,
            the_conversation=self.the_conversation,
            functions=functions,
            temperature=temperature,
            tool_output_callback=tool_output_callback,
            max_tokens=max_tokens,
            additional_parameters=additional_parameters,
            **kwargs
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
            transcript = adapter.voice_to_text(audio_file, response_format, language)
            return transcript
        elif provider.lower() == 'speechmatics':
            adapter =  self._lazy_initialization_of_adapter('SpeechmaticsAdapter')
            language = kwargs.get('language', 'en')
            transcription_config = kwargs.get('transcription_config', None)
            transcript = adapter.voice_to_text(('audio_file.'+audio_format, audio_file), language, transcription_config)
            return transcript
        elif provider.lower() == 'elevenlabs':
            adapter =  self._lazy_initialization_of_adapter('ElenenlabsAdapter')
            language = kwargs.get('language', 'eng')
            diarized = kwargs.get('diarized', True)
            transcript = adapter.voice_to_text(audio_file, language, diarized)
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
                - size (str, optional): Supported values are '256x256', '512x512', '1024x1024', '1024x1792', and '1792x1024'.
                - quality (str, optional): Supported values are 'standard' and 'hd'.

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
        else: 
            raise ValueError(f"Provider {provider} is not supported. I understand only 'openai' or 'google' as providers. ")

    def get_models(self, adapter_name: str) -> List[str]:

        adapter = self._lazy_initialization_of_adapter(adapter_name)

        models = adapter.get_models()
        return models
