import json
import os
from typing import Any, Callable, Dict, List

from llm_platform.services.conversation import Conversation, FunctionCall, FunctionResponse, Message
from llm_platform.tools.base import BaseTool
from llm_platform.adapters.serializers import function_call_from_openai_chat
from llm_platform.types import AdditionalParameters

from .openai_compatible_adapter import OpenAICompatibleAdapter


class ZaiAdapter(OpenAICompatibleAdapter):
    """Z.AI adapter (GLM models) built on the official ``zai-sdk`` ``ZaiClient``.

    ``ZaiClient`` exposes the same OpenAI-compatible ``chat.completions.create``
    surface as the shared base, so plain text chat is inherited unchanged. On
    top of it this adapter adds the two tool features Z.AI exposes through the
    standard Chat Completions ``tools`` array:

    * **Function calling** — a recursive tool-use loop (request -> execute local
      tools -> re-ask) mirroring the other tool-capable adapters.
    * **Web search** — Z.AI's built-in server-side ``web_search`` tool, enabled
      via the ``web_search`` additional parameter.

    Both tool kinds coexist: the built-in web-search tool is merged with any
    declared function tools on the same request.
    """

    BASE_URL = "https://api.z.ai/api/paas/v4/"
    ENV_VAR = "ZAI_API_KEY"

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
        return request_params

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

        # Plain chat (web search, if requested, is injected via _build_request_params).
        return super().request_llm(
            model=model,
            the_conversation=the_conversation,
            additional_parameters=additional_parameters,
        )

    def request_llm_with_functions(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool | Callable],
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        additional_parameters = self._merge_additional_parameters(additional_parameters, kwargs)

        tools = [self._convert_function_to_tool(function) for function in functions]

        request_params = self._build_request_params(model, additional_parameters)
        # Merge declared function tools with any built-in tools (e.g. web search).
        request_params["tools"] = request_params.get("tools", []) + tools
        request_params["tool_choice"] = "auto"

        history, history_kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model)
        request_params.update(history_kwargs)

        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            **request_params,
        )
        usage = self._build_usage(getattr(response, "usage", None), model)
        assistant_message = response.choices[0].message

        # No tool calls -> final answer; record it and finish.
        if not getattr(assistant_message, "tool_calls", None):
            message = Message(role="assistant", content=assistant_message.content, usage=usage)
            the_conversation.messages.append(message)
            return message

        function_calls = [
            function_call_from_openai_chat(tool_call) for tool_call in assistant_message.tool_calls
        ]
        function_responses = self._execute_tool_calls(
            function_calls, functions, tools, tool_output_callback
        )

        the_conversation.messages.append(
            Message(
                role="assistant",
                content=assistant_message.content or "",
                function_calls=function_calls,
                function_responses=function_responses,
                usage=usage,
            )
        )

        # Re-ask with the tool results appended until the model stops calling tools.
        return self.request_llm_with_functions(
            model=model,
            the_conversation=the_conversation,
            functions=functions,
            tool_output_callback=tool_output_callback,
            additional_parameters=additional_parameters,
        )
