# <document name="openai_adapter.py">
"""
Adapter for interacting with the OpenAI API.

This module provides a class, OpenAIAdapter, that conforms to the AdapterBase
interface and handles the specifics of formatting requests and parsing responses
for the OpenAI platform. It supports standard chat completions, function calling,
image generation, and audio transcription.
"""

import asyncio
import inspect
import json
from typing import Callable, Dict, List, Tuple, Union

from openai import AsyncOpenAI, OpenAI

from llm_platform.helpers.model_config import ModelConfig
from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         ExcelDocumentFile, ImageFile,
                                         MediaFile, PDFDocumentFile,
                                         TextDocumentFile, VideoFile)
from llm_platform.tools.base import BaseTool

from .adapter_base import AdapterBase

# Constants for response types
TEXT_INPUT_TYPE = "input_text"
TEXT_OUTPUT_TYPE = "output_text"
IMAGE_INPUT_TYPE = "input_image"
AUDIO_INPUT_TYPE = "input_audio"
FILE_INPUT_TYPE = "input_file"
SUMMARY_TEXT_TYPE = "summary_text"
FUNCTION_CALL_TYPE = "function_call"
MESSAGE_CALL_TYPE = "message"
IMAGE_GENERATION_CALL_TYPE = "image_generation_call"
REASONING_CALL_TYPE = "reasoning"

# Constants for transcription models
WHISPER_1 = "whisper-1"
IMAGE_MODEL = "gpt-image-1"
GPT4O_TRANSCRIBE = "gpt-4o-transcribe"
GPT4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class OpenAIAdapter(AdapterBase):
    """
    An adapter to interact with OpenAI's large language models.

    This class handles the translation between the platform's generic data
    structures (like Conversation and Message) and the format required by
    the OpenAI API. It supports both synchronous and asynchronous requests.
    """

    def __init__(self):
        """
        Initializes the OpenAIAdapter with synchronous and asynchronous clients.
        """
        super().__init__()
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.model_config = ModelConfig()

    def _convert_file_to_content(self, file: BaseFile) -> Dict | None:
        """
        Converts a BaseFile object into an OpenAI-compatible content dictionary.

        Args:
            file: The file object to convert.

        Returns:
            A dictionary representing the file content for the API, or None if
            the file type is unsupported.
        """
        if isinstance(file, ImageFile):
            return {
                "type": IMAGE_INPUT_TYPE,
                "image_url": f"data:image/{file.extension};base64,{file.base64}",
            }
        if isinstance(file, AudioFile):
            return {
                "type": AUDIO_INPUT_TYPE,
                "input_audio": {"data": file.base64, "format": "mp3"},
            }
        if isinstance(file, PDFDocumentFile):
            # Small PDFs can be uploaded directly as files
            if file.size < 32_000_000 and file.number_of_pages < 100:
                return {
                    "type": FILE_INPUT_TYPE,
                    "filename": file.name,
                    "file_data": f"data:application/pdf;base64,{file.base64}",
                }
            # For larger PDFs, provide the extracted text content
            return {
                "type": TEXT_INPUT_TYPE,
                "text": f'<document name="{file.name}">{file.text}</document>',
            }
        if isinstance(file, (TextDocumentFile, ExcelDocumentFile)):
            return {
                "type": TEXT_INPUT_TYPE,
                "text": f'<document name="{file.name}">{file.text}</document>',
            }

        self.logger.warning(f"Unsupported file type for conversion: {type(file)}. Skipping file.")
        return None

    def convert_conversation_history_to_adapter_format(
        self, conversation_messages: List[Message], **kwargs
    ) -> Tuple[List[Dict], Dict]:
        """
        Converts the platform's Conversation object into the list format
        required by the OpenAI API.

        Args:
            the_conversation: The conversation object to convert.
            model: The model being targeted (unused in this implementation but
                   part of the signature).
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the list of messages for the API and the passed-in kwargs.
        """
        history = []
        for message in conversation_messages:

            # Add any reasoning responses
            if message.thinking_responses:
                history.extend(tr.to_openai() for tr in message.thinking_responses)

            # Add any function calls
            if message.function_calls:
                history.extend(fc.to_openai() for fc in message.function_calls)  

            # Add text content        
            content_items = []
            # Add text content if it exists
            if message.content and message.content.strip() != "":
                content_items.append(
                    {
                        "type": TEXT_INPUT_TYPE if message.role == "user" else TEXT_OUTPUT_TYPE,
                        "text": message.content,
                    }
                )

            # Add file content
            if message.files:
                for file in message.files:
                    if file_content := self._convert_file_to_content(file):
                        # Prepend document text to give it context priority
                        if file_content["type"] == TEXT_INPUT_TYPE and "<document" in file_content["text"]:
                            content_items.insert(0, file_content)
                        else:
                            content_items.append(file_content)

            if content_items:
                history.append({"role": message.role, "content": content_items})

            # Add any function responses that followed the main content
            if message.function_responses:
                history.extend(fr.to_openai() for fr in message.function_responses)

        return history, kwargs

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: Dict = None,
        use_previous_response_id: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Constructs the dictionary of parameters for an OpenAI API call.

        Args:
            model: The model identifier.
            the_conversation: The current conversation state.
            additional_parameters: A dictionary of extra parameters like 'web_search'.
            use_previous_response_id: Flag to enable delta-based conversation updates.
            **kwargs: Additional keyword arguments to pass to the API.

        Returns:
            A dictionary of parameters ready for the OpenAI client.
        """
        if additional_parameters is None:
            additional_parameters = {}

        # For reasoning-focused models, certain parameters are not needed
        model_object = self.model_config[model]
        if model_object and model_object["reasoning_effort"] == 1:
            kwargs.pop("max_tokens", None)
            kwargs.pop("temperature", None)

        if 'gpt-5' in model:
            kwargs.pop("temperature", None)

        # Rename 'max_tokens' to OpenAI's expected 'max_output_tokens'
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        tools = []
        if additional_parameters.get("web_search"):
            tools.append({"type": "web_search_preview"})
        if "image" in additional_parameters.get("response_modalities", []):
            tools.append({"type": "image_generation", "quality": "high", "size": "1536x1024"})

        messages, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation.messages, **kwargs)

        parameters = {
            "model": model,
            "instructions": the_conversation.system_prompt,
            "input": messages,
            "tools": tools,
        }
        parameters.update(kwargs)

        if "reasoning" in parameters:
            parameters["reasoning"]["summary"] = "auto"

        # Use previous_response_id for more efficient, stateful conversations
        if use_previous_response_id and (prev_id := the_conversation.previous_response_id_for_openai):
            parameters["previous_response_id"] = prev_id
            # When using a previous ID, only the latest user message is needed
            last_user_message = next(
                (msg for msg in reversed(the_conversation.messages) if msg.role == "user"), None
            )
            if last_user_message:
                messages, kwargs = self.convert_conversation_history_to_adapter_format([last_user_message], **kwargs)
                parameters["input"] = messages

        return parameters

    def _parse_response(self, response) -> Tuple[str, List[ThinkingResponse], List[MediaFile], Dict]:
        """
        Parses the raw OpenAI response into structured data.

        Args:
            response: The response object from the OpenAI client.

        Returns:
            A tuple containing the answer text, a list of thinking responses,
            a list of generated media files, and a usage dictionary.
        """
        usage = {
            "model": getattr(response, "model", "Unknown model"),
            "completion_tokens": response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
        }

        outputs = getattr(response, "output", [])

        answer_text = ""
        files_from_response = []
        thinking_responses = []

        for output in outputs:
            output_type = getattr(output, "type", "")

            # Extract text from message outputs
            if output_type == MESSAGE_CALL_TYPE:
                text_parts = [
                    content.text
                    for content in (getattr(output, "content", []) or [])
                    if getattr(content, "type", "") == TEXT_OUTPUT_TYPE
                ]
                answer_text += "\n".join(text_parts)

            # Extract image output from message outputs
            if output_type == IMAGE_GENERATION_CALL_TYPE:
                image_b64_data = output.result
                files_from_response.append(
                    ImageFile.from_base64(base64_str=b64, file_name=f"image_{i}.png")
                    for i, b64 in enumerate(image_b64_data)
                )

            # Extract reasoning output from message outputs
            if output_type == REASONING_CALL_TYPE:
                summary_text = ""
                for summary in (getattr(output, "summary", []) or []):
                    if getattr(summary, "type", "") == SUMMARY_TEXT_TYPE:
                        summary_text += "\n" + summary.text

                thinking_responses.append(
                    ThinkingResponse(content=summary_text, id=output.id)
                )

        return answer_text, thinking_responses, files_from_response, usage

    def _get_function_calls_from_response(self, response) -> List[FunctionCall]:
        """Extracts function call records from an OpenAI response."""
        return [
            FunctionCall.from_openai(output)
            for output in getattr(response, "output", [])
            if getattr(output, "type", "") == FUNCTION_CALL_TYPE
        ]

    def _append_tool_messages_to_conversation(
        self,
        the_conversation: Conversation,
        response_id: str,
        usage: Dict,
        assistant_message_text: str,
        function_calls: List[FunctionCall],
        function_responses: List[FunctionResponse],
        thinking_responses: List[ThinkingResponse],
        files_from_response: List[MediaFile],
    ):
        """Appends the assistant's function call and the user's function response messages to the conversation."""
        # 1. Assistant's message with the function call request
        assistant_message = Message(
            role="assistant",
            id=response_id,
            usage=usage,
            content=assistant_message_text or "",
            function_calls=function_calls,
            thinking_responses=thinking_responses,
            files=files_from_response,
        )
        the_conversation.messages.append(assistant_message)

        # 2. User's message containing the function execution results
        user_response_message = Message(
            role="user", content="", function_responses=function_responses
        )
        the_conversation.messages.append(user_response_message)

    def _execute_tool_calls(
        self,
        function_calls: List[FunctionCall],
        functions: List[Union[BaseTool, Callable]],
        tools: List[Dict],
        tool_output_callback: Callable = None,
    ) -> List[FunctionResponse]:
        """Finds and executes synchronous tool calls."""
        response_records = []
        for tool_call in function_calls:
            function_name = tool_call.name
            arguments = json.loads(tool_call.arguments)

            try:
                # Find the corresponding function object using the tool definition name
                function_to_call = next(
                    f for i, f in enumerate(functions) if tools[i]["name"] == function_name
                )
            except StopIteration:
                raise ValueError(f"Function '{function_name}' not found in provided tools.")

            # Execute the function with keyword arguments for robustness
            response_content = function_to_call(**arguments)

            response_records.append(
                FunctionResponse(
                    name=function_name, call_id=tool_call.call_id, response=response_content
                )
            )

            if tool_output_callback:
                tool_output_callback(function_name, arguments, response_content)

        return response_records

    async def _execute_tool_calls_async(
        self,
        function_calls: List[FunctionCall],
        functions: List[Union[BaseTool, Callable]],
        tools: List[Dict],
        tool_output_callback: Callable = None,
    ) -> List[FunctionResponse]:
        """Finds and executes tool calls, handling both sync and async functions."""

        async def execute_single_call(tool_call: FunctionCall) -> FunctionResponse:
            function_name = tool_call.name
            arguments = json.loads(tool_call.arguments)

            try:
                function_to_call = next(
                    f for i, f in enumerate(functions) if tools[i]["name"] == function_name
                )
            except StopIteration:
                raise ValueError(f"Function '{function_name}' not found in provided tools.")

            if inspect.iscoroutinefunction(function_to_call):
                response_content = await function_to_call(**arguments)
            else:
                response_content = function_to_call(**arguments)

            if tool_output_callback:
                # Note: Assumes callback is synchronous. If it can be async,
                # this would need further enhancement.
                tool_output_callback(function_name, arguments, response_content)

            return FunctionResponse(
                name=function_name, call_id=tool_call.call_id, response=response_content
            )

        # Execute all tool calls concurrently
        tasks = [execute_single_call(tc) for tc in function_calls]
        return await asyncio.gather(*tasks)

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: Dict = None,
        **kwargs,
    ) -> Message:
        """
        Requests a response from an OpenAI LLM, handling standard chat and
        function calling (tool use).

        This method orchestrates sending a request, processing the response,
        and updating the conversation. If tools are provided, it will handle
        the multi-step tool-use conversation flow.

        Args:
            model: The identifier of the LLM model to use.
            the_conversation: The conversation object containing message history.
            functions: A list of tools/functions the model can call.
            tool_output_callback: A callback executed after each tool call.
            additional_parameters: Extra parameters for the API call.
            **kwargs: Additional keyword arguments for the OpenAI client.

        Returns:
            The final assistant Message object after all processing.
        """
        if functions:
            response = self.request_llm_with_functions(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
                **kwargs,
            )
        else:
            parameters = self._create_parameters_for_calling_llm(
                model, the_conversation, additional_parameters, use_previous_response_id=True, **kwargs
            )

            response = self.client.responses.create(**parameters)

        answer_text, thinking_responses, files_from_response, usage = self._parse_response(response)

        message = Message(
            role="assistant",
            id=response.id,
            usage=usage,
            content=answer_text,
            thinking_responses=thinking_responses,
            files=files_from_response,
        )
        the_conversation.messages.append(message)
        return message

    async def request_llm_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: Dict = None,
        **kwargs,
    ) -> Message:
        """
        Asynchronously requests a response from an OpenAI LLM.

        This method is the asynchronous counterpart to `request_llm`.

        Args:
            model: The identifier of the LLM model to use.
            the_conversation: The conversation object containing message history.
            functions: A list of tools/functions the model can call.
            tool_output_callback: A callback executed after each tool call.
            additional_parameters: Extra parameters for the API call.
            **kwargs: Additional keyword arguments for the OpenAI client.

        Returns:
            The final assistant Message object after all processing.
        """
        if functions:
            response = await self.request_llm_with_functions_async(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
                **kwargs,
            )
        else:
            parameters = self._create_parameters_for_calling_llm(
                model, the_conversation, additional_parameters, use_previous_response_id=True, **kwargs
            )
            response = await self.async_client.responses.create(**parameters)

        answer_text, thinking_responses, files_from_response, usage = self._parse_response(response)

        message = Message(
            role="assistant",
            id=response.id,
            usage=usage,
            content=answer_text,
            thinking_responses=thinking_responses,
            files=files_from_response,
        )
        the_conversation.messages.append(message)
        return message

    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[Union[BaseTool, Callable]],
        tool_output_callback: Callable = None,
        additional_parameters: Dict = None,
        **kwargs,
    ):
        """Handles the synchronous, recursive logic for tool-use conversations."""
        tools = [self._convert_function_to_tool(f) for f in functions]
        parameters = self._create_parameters_for_calling_llm(
            model, the_conversation, additional_parameters, use_previous_response_id=True, **kwargs
        )
        parameters["tools"].extend(tools)

        # 1. Get response from the model
        response = self.client.responses.create(**parameters)
        text, thinking, files, usage = self._parse_response(response)
        function_calls = self._get_function_calls_from_response(response)

        # 2. If no tools were called, the conversation is over.
        if not function_calls:
            return response

        # 3. Execute the requested tools
        function_responses = self._execute_tool_calls(
            function_calls, functions, tools, tool_output_callback
        )

        # 4. Append the assistant's call and the tool results to the history
        self._append_tool_messages_to_conversation(
            the_conversation, response.id, usage, text, function_calls, function_responses, thinking, files
        )

        # 5. Call the model again with the updated history to get the final answer
        return self.request_llm_with_functions(
            model, the_conversation, functions, tool_output_callback, additional_parameters, **kwargs
        )

    async def request_llm_with_functions_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[Union[BaseTool, Callable]],
        tool_output_callback: Callable = None,
        additional_parameters: Dict = None,
        **kwargs,
    ):
        """Handles the asynchronous, recursive logic for tool-use conversations."""
        tools = [self._convert_function_to_tool(f) for f in functions]
        parameters = self._create_parameters_for_calling_llm(
            model, the_conversation, additional_parameters, use_previous_response_id=True, **kwargs
        )
        parameters["tools"].extend(tools)

        # 1. Get response from the model
        response = await self.async_client.responses.create(**parameters)
        text, thinking, files, usage = self._parse_response(response)
        function_calls = self._get_function_calls_from_response(response)

        # 2. If no tools were called, the conversation is over.
        if not function_calls:
            return response

        # 3. Execute the requested tools
        function_responses = await self._execute_tool_calls_async(
            function_calls, functions, tools, tool_output_callback
        )

        # 4. Append the assistant's call and the tool results to the history
        self._append_tool_messages_to_conversation(
            the_conversation, response.id, usage, text, function_calls, function_responses, thinking, files
        )

        # 5. Call the model again with the updated history to get the final answer
        return await self.request_llm_with_functions_async(
            model, the_conversation, functions, tool_output_callback, additional_parameters, **kwargs
        )

    def generate_image(self, prompt: str, n: int = 1, **kwargs) -> List[ImageFile]:
        """
        Generates images from a text prompt using gpt-image-1.

        Args:
            prompt: The text description of the desired image(s).
            n: The number of images to generate.
            **kwargs: Additional parameters for the OpenAI API like 'size', 'quality', etc.

        Returns:
            A list of ImageFile objects.
        """
        # Start with default parameters
        params = {
            "size": "1024x1536",
            "quality": "high",
            "output_format": "png",
        }

        # Update the defaults with any user-provided kwargs
        # This allows users to override 'size', 'quality', etc.
        params.update(kwargs)

        response = self.client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            n=n,
            **params,  # Unpack the combined parameters
        )

        return [
            ImageFile.from_base64(base64_str=img.b64_json, file_name=f"generated_image_{i}.png")
            for i, img in enumerate(response.data)
        ]

    def edit_image(self, prompt: str, images: List[ImageFile], n: int = 1, **kwargs) -> List[ImageFile]:
        """
        Edits existing images based on a text prompt.

        Args:
            prompt: The text description of the edits to make.
            images: A list of ImageFile objects to edit.
            n: The number of edited versions to generate per input image.
            **kwargs: Additional parameters like 'size'.

        Returns:
            A list of the edited ImageFile objects.
        """
        response = self.client.images.edit(
            model=IMAGE_MODEL,
            prompt=prompt,
            image=[image.bytes_io for image in images],
            n=n,
            size=kwargs.pop("size", "1024x1536"),
            **kwargs,
        )
        return [
            ImageFile.from_base64(base64_str=img.b64_json, file_name=f"edited_image_{i}.png")
            for i, img in enumerate(response.data)
        ]

    def voice_to_text(
        self, audio_file, response_format: str = "text", language: str = "en", model: str = GPT4O_TRANSCRIBE
    ):
        """
        Transcribes an audio file to text using a Whisper model.

        Args:
            audio_file: The audio file object to transcribe.
            response_format: The desired output format ('text', 'srt', 'verbose_json').
            language: The language of the audio in ISO-639-1 format.
            model: The transcription model to use.

        Returns:
            The transcription result in the specified format.
        """
        VALID_FORMATS = ["text", "srt", "verbose_json"]
        VALID_MODELS = [GPT4O_TRANSCRIBE, GPT4O_MINI_TRANSCRIBE, WHISPER_1]
        if response_format not in VALID_FORMATS:
            raise ValueError(f"Invalid response_format. Must be one of {VALID_FORMATS}")
        if model not in VALID_MODELS:
            raise ValueError(f"Invalid model. Must be one of {VALID_MODELS}")

        return self.client.audio.transcriptions.create(
            model=model, file=audio_file, response_format=response_format, language=language
        )

    def _convert_func_to_tool(self, func: Callable) -> Dict:
        """Converts a Python function to an OpenAI tool definition using inspection."""
        sig = inspect.signature(func)
        TYPE_MAP = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        properties = {}
        required = []
        for name, param in sig.parameters.items():
            # Default to 'string' if type is missing or not in our map
            schema_type = TYPE_MAP.get(param.annotation, "string")
            properties[name] = {"type": schema_type, "description": ""}  # Description can be enhanced
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "name": func.__name__,
            "description": func.__doc__ or f"Executes the {func.__name__} function.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _convert_function_to_tool(self, func: Union[BaseTool, Callable]) -> Dict:
        """
        Converts a BaseTool or a standard Python function into an OpenAI
        tool definition.

        Args:
            func: The function or BaseTool instance to convert.

        Returns:
            A dictionary representing the OpenAI tool.

        Raises:
            TypeError: If the input is not a BaseTool or a callable function.
        """
        if isinstance(func, BaseTool):
            tool = func.to_params(provider="openai")
            tool["type"] = "function"
        elif callable(func):
            tool = self._convert_func_to_tool(func)
        else:
            raise TypeError("func must be either a BaseTool instance or a callable function")
        return tool

    def get_models(self) -> List[str]:
        """
        Retrieves a list of available model IDs from OpenAI.

        Returns:
            A list of string identifiers for the available models.
        """
        models = self.client.models.list()
        return [model.id for model in models.data]