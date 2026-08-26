import json
import os
from typing import Any, Callable, Dict, List

from llm_platform.services.conversation import Conversation, FunctionCall, FunctionResponse, Message
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import function_call_from_openai_chat
from llm_platform.types import AdditionalParameters

from .adapter_base import AdapterBase, MAX_TOOL_ROUNDS
from .openai_compatible_adapter import OpenAICompatibleAdapter


class ZaiAdapter(OpenAICompatibleAdapter):
    """Z.AI adapter (GLM models) built on the official ``zai-sdk`` ``ZaiClient``.

    ``ZaiClient`` exposes the same OpenAI-compatible ``chat.completions.create``
    surface as the shared base. This adapter reuses that common serialization
    and adds Z.AI's provider-specific thinking, tool, and structured-output
    behavior:

    * **Function calling** — a recursive tool-use loop (request -> execute local
      tools -> re-ask) mirroring the other tool-capable adapters.
    * **Web search** — Z.AI's built-in server-side ``web_search`` tool, enabled
      via the ``web_search`` additional parameter.
    * **Preserved thinking** — prior ``reasoning_content`` is replayed exactly
      so interleaved thinking remains coherent across tool rounds and turns.
    * **Structured output** — Z.AI JSON mode is enabled with
      ``response_format={"type": "json_object"}``; Pydantic/JSON schemas are
      added to the system instruction because the API does not accept a schema
      inside ``response_format``.

    Both tool kinds coexist: the built-in web-search tool is merged with any
    declared function tools on the same request.
    """

    BASE_URL = "https://api.z.ai/api/paas/v4/"
    ENV_VAR = "ZAI_API_KEY"

    # The base's native async path is backed by AsyncOpenAI and has no tool
    # calling — it would bypass the official ZaiClient and regress this
    # adapter's async function-calling support. Keep the thread-offloaded
    # AdapterBase default, which runs the full sync `request_llm` off the loop.
    request_llm_async = AdapterBase.request_llm_async

    def _build_client(self):
        from zai import ZaiClient
        return ZaiClient(api_key=os.getenv(self.ENV_VAR), base_url=self.BASE_URL)

    def _build_builtin_tools(self, additional_parameters: AdditionalParameters) -> List[Dict]:
        """Z.AI server-side tools requested through ``additional_parameters``.

        Currently only web search: enabling it lets GLM derive search queries
        from the conversation automatically (no static query is forced) and
        return the retrieved results alongside the answer.
        """
        tools: List[Dict] = []
        if additional_parameters.get("web_search"):
            tools.append(
                {
                    "type": "web_search",
                    "web_search": {
                        "enable": True,
                        "search_engine": "search-prime",
                        "search_result": True,
                    },
                }
            )
        return tools

    def _build_request_params(self, model: str, additional_parameters: AdditionalParameters) -> Dict[str, Any]:
        # Reuse the shared OpenAI-compatible marshalling, then attach any
        # built-in (server-side) tools so both the plain and tool-calling paths
        # pick up web search uniformly.
        request_params = super()._build_request_params(model, additional_parameters)
        builtin_tools = self._build_builtin_tools(additional_parameters)
        if builtin_tools:
            request_params["tools"] = builtin_tools

        if additional_parameters.get("structured_output"):
            request_params["response_format"] = {"type": "json_object"}

        return request_params

    @staticmethod
    def _structured_output_schema(structured_output: Any) -> Dict[str, Any] | None:
        """Return the requested JSON Schema, if the caller supplied one."""
        if structured_output is True:
            return None

        if hasattr(structured_output, "model_json_schema"):
            return structured_output.model_json_schema()

        if not isinstance(structured_output, dict):
            raise TypeError(
                "structured_output must be True, a Pydantic model class, "
                "or a JSON Schema dict"
            )

        if structured_output.get("type") == "json_object":
            return None
        if structured_output.get("type") == "json_schema":
            json_schema = structured_output.get("json_schema", {})
            schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
            if not isinstance(schema, dict):
                raise TypeError("structured_output json_schema must contain a schema dict")
            return schema

        return structured_output

    def convert_conversation_history_to_adapter_format(
        self,
        the_conversation: Conversation,
        model: str,
        **kwargs,
    ):
        """Serialize history and preserve Z.AI reasoning blocks verbatim."""
        history, history_kwargs = super().convert_conversation_history_to_adapter_format(
            the_conversation,
            model,
            **kwargs,
        )

        history_index = 1  # The shared serializer puts the system message first.
        for message in the_conversation.messages:
            if message.role == "assistant" and message.thinking_responses:
                history[history_index]["reasoning_content"] = "".join(
                    response.content for response in message.thinking_responses
                )
            history_index += 1 + len(message.function_responses)

        return history, history_kwargs

    def _prepare_history(
        self,
        the_conversation: Conversation,
        model: str,
        structured_output: Any = None,
    ):
        history, history_kwargs = self.convert_conversation_history_to_adapter_format(
            the_conversation,
            model,
        )
        if not structured_output:
            return history, history_kwargs

        schema = self._structured_output_schema(structured_output)
        instruction = "Return only a valid JSON object."
        if schema is not None:
            instruction += (
                " The JSON object must match this JSON Schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )

        system_content = history[0].get("content") or ""
        history[0]["content"] = (
            f"{system_content}\n\n{instruction}" if system_content else instruction
        )
        return history, history_kwargs

    def _convert_function_to_tool(self, func: BaseTool | Callable) -> Dict:
        """Convert a ``BaseTool`` or plain callable into a Chat Completions function tool."""
        if isinstance(func, BaseTool):
            schema = func.to_params(provider="openai")
        elif callable(func):
            schema = self._callable_to_json_schema(func)
        else:
            raise TypeError("func must be either a BaseTool or a callable function")
        return {"type": "function", "function": schema}

    def _execute_tool_calls(
        self,
        function_calls: List[FunctionCall],
        functions: List[BaseTool | Callable],
        tools: List[Dict],
        tool_output_callback: Callable = None,
    ) -> List[FunctionResponse]:
        tool_map = {tool["function"]["name"]: func for tool, func in zip(tools, functions)}

        function_responses = []
        for function_call in function_calls:
            function_to_call = tool_map.get(function_call.name)
            if function_to_call is None:
                raise ValueError(f"Function {function_call.name} not found in tools")

            arguments = json.loads(function_call.arguments)
            response = function_to_call(**arguments)

            function_responses.append(
                FunctionResponse(
                    name=function_call.name,
                    call_id=function_call.call_id,
                    response=response,
                )
            )

            if tool_output_callback:
                tool_output_callback(function_call.name, arguments, response)

        return function_responses

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        if functions:
            return self.request_llm_with_functions(
                model=model,
                the_conversation=the_conversation,
                functions=functions,
                tool_output_callback=tool_output_callback,
                additional_parameters=additional_parameters,
            )

        request_params = self._build_request_params(model, additional_parameters)
        history, history_kwargs = self._prepare_history(
            the_conversation,
            model,
            additional_parameters.get("structured_output"),
        )
        request_params.update(history_kwargs)

        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            **request_params,
        )
        message = self._message_from_response(model, response)
        the_conversation.messages.append(message)
        return message

    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        _tool_round: int = 0,
        **kwargs,
    ) -> Message:
        if _tool_round >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Exceeded maximum tool-calling rounds ({MAX_TOOL_ROUNDS}) for model {model}"
            )

        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        tools = [self._convert_function_to_tool(function) for function in functions]

        request_params = self._build_request_params(model, additional_parameters)
        # Merge declared function tools with any built-in tools (e.g. web search).
        request_params["tools"] = request_params.get("tools", []) + tools
        request_params["tool_choice"] = "auto"

        history, history_kwargs = self._prepare_history(
            the_conversation,
            model,
            additional_parameters.get("structured_output"),
        )
        request_params.update(history_kwargs)

        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            **request_params,
        )
        assistant_message = response.choices[0].message

        # No tool calls -> final answer; record it and finish.
        if not getattr(assistant_message, "tool_calls", None):
            message = self._message_from_response(model, response)
            the_conversation.messages.append(message)
            return message

        function_calls = [
            function_call_from_openai_chat(tool_call) for tool_call in assistant_message.tool_calls
        ]
        function_responses = self._execute_tool_calls(
            function_calls, functions, tools, tool_output_callback
        )

        message = self._message_from_response(model, response)
        message.content = assistant_message.content or ""
        message.function_calls = function_calls
        message.function_responses = function_responses
        the_conversation.messages.append(message)

        # Re-ask with the tool results appended until the model stops calling tools.
        return self.request_llm_with_functions(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
            _tool_round=_tool_round + 1,
        )
