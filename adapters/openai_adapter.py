# <document name="openai_adapter.py">
"""
Adapter for interacting with the OpenAI API.

This module provides a class, OpenAIAdapter, that conforms to the AdapterBase
interface and handles the specifics of formatting requests and parsing responses
for the OpenAI platform. It supports standard chat completions and function calling.
"""

import asyncio
import inspect
import json
from typing import Callable, Dict, List, Tuple, Union
from loguru import logger
import time

from llm_platform.services.conversation import (Conversation, FunctionCall,
                                                FunctionResponse, Message,
                                                ThinkingResponse)
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile, define_file_type)
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import (
    function_call_from_openai,
    function_call_to_openai,
    function_response_to_openai,
    thinking_response_to_openai,
)
from llm_platform.types import AdditionalParameters

from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS, PDF_INLINE_MAX_BYTES, PDF_INLINE_MAX_PAGES

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
CODE_INTERPRETER_CALL_TYPE = "code_interpreter_call"


class OpenAIAdapter(AdapterBase):
    """
    An adapter to interact with OpenAI's large language models.

    This class handles the translation between the platform's generic data
    structures (like Conversation and Message) and the format required by
    the OpenAI API. It supports both synchronous and asynchronous requests.
    """

    def __init__(self):
        super().__init__()
        self._async_client = None

    def _build_client(self):
        from openai import OpenAI
        return OpenAI()

    @property
    def async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI()
        return self._async_client

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
                "image_url": self._image_data_url(file),
            }
        if isinstance(file, AudioFile):
            return {
                "type": AUDIO_INPUT_TYPE,
                "input_audio": {"data": file.base64, "format": "mp3"},
            }
        if isinstance(file, PDFDocumentFile):
            # Small PDFs can be uploaded directly as files
            if file.size < PDF_INLINE_MAX_BYTES and file.number_of_pages < PDF_INLINE_MAX_PAGES:
                return {
                    "type": FILE_INPUT_TYPE,
                    "filename": file.name,
                    "file_data": f"data:application/pdf;base64,{file.base64}",
                }
            # For larger PDFs, provide the extracted text content
            return {
                "type": TEXT_INPUT_TYPE,
                "text": self._document_xml(file),
            }
        if isinstance(file, (TextDocumentFile, ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
            return {
                "type": TEXT_INPUT_TYPE,
                "text": self._document_xml(file),
            }

        logger.warning(f"Unsupported file type for conversion: {type(file)}. Skipping file.")
        return None

    def convert_conversation_history_to_adapter_format(
        self, conversation_messages: List[Message]
    ) -> List[Dict]:
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
                history.extend(thinking_response_to_openai(tr) for tr in message.thinking_responses)

            # Add any function calls
            if message.function_calls:
                history.extend(function_call_to_openai(fc) for fc in message.function_calls)  

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
                history.extend(function_response_to_openai(fr) for fr in message.function_responses)

        return history

    def _create_parameters_for_calling_llm(
        self,
        model: str,
        the_conversation: Conversation,
        additional_parameters: AdditionalParameters | None = None,
        use_previous_response_id: bool = True,
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

        model_object = self.model_config[model]

        tools = []
        if additional_parameters.get("web_search"):
            tools.append({"type": "web_search_preview"})
        if additional_parameters.get("code_execution"):
            tools.append({
                "type": "code_interpreter",
                "container": {"type": "auto"}
            })

        messages = self.convert_conversation_history_to_adapter_format(the_conversation.messages)

        parameters = {
            "model": model,
            "instructions": the_conversation.system_prompt,
            "input": messages,
            "tools": tools,
        }

        passthrough_keys = {
            "web_search",
            "code_execution",
            "response_modalities",
            "structured_output",
        }
        for key, value in additional_parameters.items():
            if key in passthrough_keys:
                continue
            parameters[key] = value

        if "reasoning" in parameters:
            parameters["reasoning"]["summary"] = "auto"

        # Use previous_response_id for more efficient, stateful conversations
        if use_previous_response_id and (prev_id := the_conversation.last_assistant_id):
            parameters["previous_response_id"] = prev_id
            # When using a previous ID, only the latest user message is needed
            last_user_message = next(
                (msg for msg in reversed(the_conversation.messages) if msg.role == "user"), None
            )
            if last_user_message:
                messages = self.convert_conversation_history_to_adapter_format([last_user_message])
                parameters["input"] = messages

        # Use background mode for long-running tasks if supported
        if model_object and model_object["background_mode"]:
            parameters["background"] = True

        # Structured output
        if structured_output_class := additional_parameters.get("structured_output", None):
            parameters["text_format"] = structured_output_class

        return parameters

    def _parse_response(self, response) -> Tuple[str, List[ThinkingResponse], List[MediaFile], List[Dict], Dict]:
        """
        Parses the raw OpenAI response into structured data.

        This is pure: it performs no network IO. Container-file citations are
        returned as metadata for the caller to fetch via
        `_retrieve_container_files` / `_retrieve_container_files_async`.

        Args:
            response: The response object from the OpenAI client.

        Returns:
            * the answer text,
            * a list of thinking responses,
            * a list of generated media files,
            * a list of container-file citation dicts (to be retrieved separately), and
            * a usage dictionary.
        """

        usage = self._build_usage(
            getattr(response, "usage", None),
            getattr(response, "model", "Unknown model"),
            completion_attr="output_tokens",
            prompt_attr="input_tokens",
        )

        outputs = getattr(response, "output", [])

        answer_text = ""
        files_from_response = []
        container_file_citations = []
        thinking_responses = []

        for output in outputs:
            output_type = getattr(output, "type", "")

            # Extract text from message outputs
            if output_type == MESSAGE_CALL_TYPE:
                for content in (getattr(output, "content", []) or []):
                    if getattr(content, "type", "") == TEXT_OUTPUT_TYPE:
                        if answer_text:
                            answer_text += "\n"
                        answer_text += content.text

                    # Collect container-file citations as metadata only. The bytes
                    # are fetched separately (see `_retrieve_container_files`) so that
                    # parsing stays pure and the async path does not block on network IO.
                    for annotation in (getattr(content, "annotations", []) or []):
                        if getattr(annotation, "type", "") == "container_file_citation":
                            container_file_citations.append({
                                "container_id": annotation.container_id,
                                "file_id": annotation.file_id,
                                "filename": annotation.filename,
                            })


            # Extract image output from message outputs. `result` is a single
            # base64-encoded image string, not a list of images.
            if output_type == IMAGE_GENERATION_CALL_TYPE:
                if image_b64 := getattr(output, "result", None):
                    files_from_response.append(
                        ImageFile.from_base64(
                            base64_str=image_b64,
                            file_name=f"image_{len(files_from_response)}.png",
                        )
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

            if output_type == CODE_INTERPRETER_CALL_TYPE:
                code_text = getattr(output, "code", "")
                answer_text += f"\n```\n{code_text}```\n"

        return answer_text, thinking_responses, files_from_response, container_file_citations, usage

    @staticmethod
    def _build_file_from_container_data(data, filename) -> MediaFile | None:
        """Build a file object from raw container-file bytes and its filename."""
        file_type = define_file_type(filename)
        if file_type == "image":
            return ImageFile.from_bytes(data, file_name=filename)
        elif file_type == "video":
            return VideoFile.from_bytes(data, file_name=filename)
        elif file_type == "audio":
            return AudioFile.from_bytes(data, file_name=filename)
        elif file_type == "text":
            return TextDocumentFile.from_string(data.decode("utf-8", errors="replace"), name=filename)
        elif file_type == "pdf":
            return PDFDocumentFile.from_bytes(data, file_name=filename)
        elif file_type == "excel":
            return ExcelDocumentFile.from_bytes(data, file_name=filename)
        elif file_type == "word":
            return WordDocumentFile.from_bytes(data, file_name=filename)
        elif file_type == "powerpoint":
            return PowerPointDocumentFile.from_bytes(data, file_name=filename)
        return None

    def _retrieve_container_files(self, citations: List[Dict]) -> List[MediaFile]:
        """Fetch container-file citations (sync). Kept out of `_parse_response`
        so parsing is pure and testable."""
        files = []
        for citation in citations:
            response = self.client.containers.files.content.retrieve(
                container_id=citation["container_id"],
                file_id=citation["file_id"],
            )
            retrieved_file = self._build_file_from_container_data(
                getattr(response, "content", None), citation["filename"]
            )
            if retrieved_file:
                files.append(retrieved_file)
        return files

    async def _retrieve_container_files_async(self, citations: List[Dict]) -> List[MediaFile]:
        """Async counterpart of `_retrieve_container_files` (uses the async client)."""
        files = []
        for citation in citations:
            response = await self.async_client.containers.files.content.retrieve(
                container_id=citation["container_id"],
                file_id=citation["file_id"],
            )
            retrieved_file = self._build_file_from_container_data(
                getattr(response, "content", None), citation["filename"]
            )
            if retrieved_file:
                files.append(retrieved_file)
        return files

    def _poll_background_response(self, response):
        """Poll a background-mode response until it leaves the queued/in-progress states."""
        while response.status in {"queued", "in_progress"}:
            time.sleep(10)  # Poll every 10 seconds
            response = self.client.responses.retrieve(response.id)
        return response

    async def _poll_background_response_async(self, response):
        """Async counterpart of `_poll_background_response` (uses the async client)."""
        while response.status in {"queued", "in_progress"}:
            await asyncio.sleep(10)  # Poll every 10 seconds
            response = await self.async_client.responses.retrieve(response.id)
        return response

    def _get_function_calls_from_response(self, response) -> List[FunctionCall]:
        """Extracts function call records from an OpenAI response."""
        return [
            function_call_from_openai(output)
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

            # Find the corresponding function object using the tool definition name
            tool_map = {tool["name"]: func for tool, func in zip(tools, functions)}
            function_to_call = tool_map.get(function_name)
            if not function_to_call:
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
        additional_parameters: AdditionalParameters | None = None,
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
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        if functions:
            response = self.request_llm_with_functions(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )
        else:
            parameters = self._create_parameters_for_calling_llm(
                model, the_conversation, additional_parameters, use_previous_response_id=True
            )

            # For Structured model outputs
            if additional_parameters.get("structured_output", None):
                response = self.client.responses.parse(**parameters)
            # For all other outputs
            else:
                response = self.client.responses.create(**parameters)

            if parameters.get("background"):
                logger.info(f"Background task initiated")
                response = self._poll_background_response(response)

        answer_text, thinking_responses, files_from_response, citations, usage = self._parse_response(response)
        # Container files come before parsed (generated) files to preserve the pre-refactor order.
        files_from_response = self._retrieve_container_files(citations) + files_from_response

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
        additional_parameters: AdditionalParameters | None = None,
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
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        if functions:
            response = await self.request_llm_with_functions_async(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )
        else:
            parameters = self._create_parameters_for_calling_llm(
                model, the_conversation, additional_parameters, use_previous_response_id=True
            )
            response = await self.async_client.responses.create(**parameters)

            if parameters.get("background"):
                logger.info(f"Background task initiated")
                response = await self._poll_background_response_async(response)

        answer_text, thinking_responses, files_from_response, citations, usage = self._parse_response(response)
        # Container files come before parsed (generated) files to preserve the pre-refactor order.
        files_from_response = await self._retrieve_container_files_async(citations) + files_from_response

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
        additional_parameters: AdditionalParameters | None = None,
        _tool_round: int = 0,
        **kwargs,
    ):
        """Handles the synchronous, recursive logic for tool-use conversations."""
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        tools = [self._convert_function_to_tool(f) for f in functions]
        parameters = self._create_parameters_for_calling_llm(
            model, the_conversation, additional_parameters, use_previous_response_id=True
        )
        parameters["tools"].extend(tools)

        # 1. Get response from the model
        if "text_format" in parameters:
            response = self.client.responses.parse(**parameters)
        else:
            response = self.client.responses.create(**parameters)

        if parameters.get("background"):
            logger.info(f"Background task initiated. Response ID: {response.id}")
            response = self._poll_background_response(response)

        text, thinking, files, citations, usage = self._parse_response(response)
        files = self._retrieve_container_files(citations) + files
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
            model, the_conversation, functions, tool_output_callback, additional_parameters,
            _tool_round=_tool_round + 1,
        )

    async def request_llm_with_functions_async(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[Union[BaseTool, Callable]],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        _tool_round: int = 0,
        **kwargs,
    ):
        """Handles the asynchronous, recursive logic for tool-use conversations."""
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        tools = [self._convert_function_to_tool(f) for f in functions]
        parameters = self._create_parameters_for_calling_llm(
            model, the_conversation, additional_parameters, use_previous_response_id=True
        )
        parameters["tools"].extend(tools)

        # 1. Get response from the model
        response = await self.async_client.responses.create(**parameters)

        if parameters.get("background"):
            logger.info(f"Background task initiated. Response ID: {response.id}")
            response = await self._poll_background_response_async(response)

        text, thinking, files, citations, usage = self._parse_response(response)
        files = await self._retrieve_container_files_async(citations) + files
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
            model, the_conversation, functions, tool_output_callback, additional_parameters,
            _tool_round=_tool_round + 1,
        )

    def _convert_func_to_tool(self, func: Callable) -> Dict:
        """Converts a Python function to an OpenAI tool definition using inspection."""
        schema = self._callable_to_json_schema(func)
        return {
            "type": "function",
            "name": schema["name"],
            "description": schema["description"] or f"Executes the {func.__name__} function.",
            "parameters": {**schema["parameters"], "additionalProperties": False},
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
