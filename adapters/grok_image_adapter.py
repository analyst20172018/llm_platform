"""
Adapter for interacting with the Grok Image model.
"""

import asyncio
import inspect
import json
from typing import Callable, Dict, List, Tuple, Union
from urllib import response
from loguru import logger
import time
import os
from io import BytesIO

from xai_sdk import Client
from xai_sdk.chat import user, system, image, assistant, tool, tool_result
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters
from xai_sdk.tools import web_search, x_search, code_execution

from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile, define_file_type)
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters

class GrokImageAdapter:
    """
    An adapter to interact with Grok's image models.
    """

    def __init__(self):
        """
        Initializes the GrokImageAdapter with synchronous and asynchronous clients.
        """
        self.client = Client(api_key = os.getenv("XAI_API_KEY"))

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters | None = None,
    ) -> Dict:
        """
        Constructs the dictionary of parameters for an Grok API call.

        Args:
            model: The model identifier.
            the_conversation: The current conversation state.
            additional_parameters: A dictionary of extra parameters like 'web_search'.
            use_previous_response_id: Flag to enable delta-based conversation updates.
            **kwargs: Additional keyword arguments to pass to the API.

        Returns:
            A dictionary of parameters ready for the Grok client.
        """
        additional_parameters = additional_parameters or {}

        parameters = {
            "model": model,
            "prompt": the_conversation.messages[-1].content,
            "image_format": "base64",
            "aspect_ratio": additional_parameters.get("aspect_ratio"),
            "resolution": additional_parameters.get("resolution"),
        }

        return parameters

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
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
            if len(image_files_in_last_message) > 1:
                logger.warning("Grok Image model only supports editing one image at a time. Using the first image provided.")

            image_file_to_edit_base64 = image_files_in_last_message[0].base64
            image_file_to_edit_extension = image_files_in_last_message[0].extension
            parameters['image_url'] = f"data:image/{image_file_to_edit_extension};base64,{image_file_to_edit_base64}"

        result = self.client.image.sample(**parameters)

        images_output: list[ImageFile] = []
        output_format = "jpeg"
        image_bytes = getattr(result, "image", None)
        if image_bytes:
            image_file = ImageFile.from_bytes(image_bytes, file_name=f"image_0.{output_format}")
            images_output.append(image_file)

        usage = {}
        if result_usage := getattr(result, "usage", None):
            usage = {
                "model": model,
                "completion_tokens": None,
                "prompt_tokens": None,
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
