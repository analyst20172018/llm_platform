"""
Adapter for interacting with the OpenAI API Image model.
"""

import asyncio
import inspect
import json
from typing import Callable, Dict, List, Tuple, Union
from loguru import logger
import time

from openai import AsyncOpenAI, OpenAI

from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile, define_file_type)
from llm_platform.tools.base import BaseTool

class OpenAIImageAdapter:
    """
    An adapter to interact with OpenAI's _image_ models.
    """

    def __init__(self):
        """
        Initializes the OpenAIImageAdapter with synchronous and asynchronous clients.
        """
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: Dict = None,
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
        if additional_parameters.get("web_search"):
            logger.warning("Web search is not supported for image models.")
        if additional_parameters.get("code_execution"):
            logger.warning("Code execution is not supported for image models.")
        if additional_parameters.get("reasoning"):
            logger.warning("Reasoning is not supported for image models.")
        if additional_parameters.get("structured_output"):
            logger.warning("Structured output is not supported for image models.")

        parameters = {
            "model": model,
            "prompt": the_conversation.messages[-1].content,
            "n": 1,
            "size": additional_parameters.get("size"),
            "background": additional_parameters.get("background"),
            "quality": additional_parameters.get("quality"),
            "output_format": additional_parameters.get("output_format"),
        }

        return parameters

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
        if additional_parameters is None:
            additional_parameters = {}

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)

        if functions:
            logger.error("Image model does not support functions.")
            message = Message(role="assistant", content="Image model does not support functions.", usage={})
            the_conversation.messages.append(message)
            return message
        
        # Create parameters for calling LLM
        parameters = self._create_parameters_for_calling_llm(
            model, the_conversation, additional_parameters
        )

        # Get image files from last message
        image_files_in_last_message = [file for file in the_conversation.messages[-1].files if isinstance(file, ImageFile)]
        if (not image_files_in_last_message) and (len(the_conversation.messages) > 1):
            # If there are no image files in the last message, then we will go to the previous assistant message and try to find the image files there
            image_files_in_last_message = [file for file in the_conversation.messages[-2].files if isinstance(file, ImageFile)]

        # If there are provided image(s), then we will use them to edit the image
        if image_files_in_last_message:

            image_files: list[BytesIO] = []
            for image_file in image_files_in_last_message:
                buf = image_file.bytes_io
                buf.name = image_file.name
                image_files.append(buf)

            result = self.client.images.edit(image=image_files, **parameters)

        # If there are no image files in the last message, then we will generate a new image
        else:
            result = self.client.images.generate(**parameters)

        output_format = getattr(result, "output_format", "png")
        images_output: list[ImageFile] = []
        for i, item in enumerate(getattr(result, "data", []) or []):
            b64 = getattr(item, "b64_json", None)
            if not b64:
                continue
            image_file = ImageFile.from_base64(b64, file_name=f"image_{i}.{output_format}")
            images_output.append(image_file)

        usage = {}
        if result_usage := getattr(result, "usage", None):
            usage = {
                "model": model,
                "completion_tokens": getattr(result_usage, "total_tokens", None),
                "prompt_tokens": getattr(result_usage, "input_tokens", None),
            }

        message = Message(
            role="assistant",
            id=None,
            usage=usage,
            content="",
            thinking_responses=[],
            files=images_output,
        )
        the_conversation.messages.append(message)
        return message