"""Schema semantics and offline provider serialization regressions."""

from __future__ import annotations

import asyncio
import copy
import functools
import json
from typing import Annotated, Literal

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.grok_adapter import GrokAdapter
from llm_platform.adapters.kimi_adapter import KimiAdapter
from llm_platform.adapters.mistral_adapter import MistralAdapter
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.zai_adapter import ZaiAdapter
from llm_platform.services.conversation import Conversation, FunctionCall, Message
from llm_platform.tools.base import BaseTool


class Employee(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    manager: str | None
    reports: list[Employee] = Field(default_factory=list)


class Options(TypedDict):
    enabled: bool
    label: NotRequired[str | None]


class InputModel(BaseModel):
    employee: Employee
    scores: dict[str, list[int | None]]
    options: Options
    limit: Annotated[int, Field(gt=0, lt=10)] = 3


class EmployeeTool(BaseTool):
    """Inspect an employee."""

    InputModel = InputModel

    def __call__(self, **kwargs):
        return kwargs


def inspect_employee(
    employee: Employee,
    scores: dict[str, list[int | None]],
    options: Options,
    *,
    limit: Annotated[int, Field(gt=0, lt=10)] = 3,
):
    """Inspect an employee."""
    return employee


def valid_arguments():
    return {
        "employee": {"name": "Alice", "manager": None, "reports": [
            {"name": "Bob", "manager": "Alice"},
        ]},
        "scores": {"team": [1, None]},
        "options": {"enabled": True},
    }


@pytest.mark.parametrize("use_model", [False, True])
def test_nested_recursive_nullable_and_typed_containers_keep_validation(use_model):
    schema = (EmployeeTool.to_params("google")["parameters"] if use_model else
              OpenAIAdapter()._callable_to_json_schema(inspect_employee)["parameters"])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = valid_arguments()
    assert validator.is_valid(valid)
    assert validator.is_valid({**valid, "options": {"enabled": True, "label": None}})
    assert set(schema["required"]) == {"employee", "scores", "options"}
    assert schema["$defs"]["Employee"]["properties"]["reports"]["items"] == {
        "$ref": "#/$defs/Employee",
    }
    for invalid in [
        {**valid, "employee": {"name": "Alice"}},  # nullable still required
        {**valid, "employee": {"name": "Alice", "manager": None, "reports": [{}]}},
        {**valid, "employee": {"name": "Alice", "manager": None, "extra": 1}},
        {**valid, "scores": {"team": ["wrong"]}},
        {**valid, "options": {}},
        {**valid, "options": {"enabled": "yes"}},
        {**valid, "limit": 0},
        {**valid, "limit": 10},
    ]:
        assert not validator.is_valid(invalid), invalid


def test_cleanup_visits_schemas_without_changing_literal_data_or_keyword_names():
    literal = {"title": "keep", "required": True, "properties": {"title": 1}}
    schema = {
        "title": "remove",
        "$defs": {"title": {"type": "object", "required": ["name"],
                              "properties": {"name": {"type": "string"}}}},
        "type": "object",
        "properties": {
            "title": {"$ref": "#/$defs/title", "description": "keep sibling"},
            "required": {"type": "string", "required": True},
            "properties": {"type": "object", "default": literal, "examples": [literal]},
        },
        "required": ["title"],
        "allOf": [{"required": ["required"]}],
    }
    original = copy.deepcopy(schema)
    cleaned = BaseTool.clean_schema(schema)
    Draft202012Validator.check_schema(cleaned)
    assert "title" not in cleaned
    assert cleaned["properties"]["title"] == schema["properties"]["title"]
    assert cleaned["$defs"]["title"]["required"] == ["name"]
    assert cleaned["allOf"] == [{"required": ["required"]}]
    assert cleaned["properties"]["required"] == {"type": "string"}
    assert cleaned["properties"]["properties"]["default"] == literal
    cleaned["properties"]["properties"]["examples"][0]["title"] = "changed"
    assert schema == original


def test_google_compatibility_helper_preserves_refs_and_constraints_without_mutation():
    original = Employee.model_json_schema()
    result = BaseTool.resolve_schema_for_google(original)
    assert result == original
    result["$defs"]["Employee"]["required"].clear()
    assert original["$defs"]["Employee"]["required"] == ["name", "manager"]


def test_callable_collections_literals_unions_and_unannotated_values():
    def containers(values: list[int], mapping: dict[str, bool], pair: tuple[str, int],
                   unique: set[int], mode: Literal["a", "b"], choice: int | str,
                   anything, bare_list: list, bare_dict: dict):
        pass

    schema = OpenAIAdapter()._callable_to_json_schema(containers)["parameters"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = dict(values=[1], mapping={"a": True}, pair=["x", 2], unique=[1, 2],
                 mode="a", choice=4, anything=None, bare_list=[None], bare_dict={"x": 1})
    assert validator.is_valid(valid)
    assert validator.is_valid({**valid, "anything": {"x": [1]}, "choice": "four"})
    for key, bad in [("values", ["1"]), ("mapping", {"a": 1}), ("pair", [1, "x"]),
                     ("unique", [1, 1]), ("mode", "c"), ("choice", None)]:
        assert not validator.is_valid({**valid, key: bad})


@pytest.mark.parametrize("signature", ["x, /", "*args", "**kwargs"])
def test_unrepresentable_signatures_fail_explicitly(signature):
    namespace = {}
    exec(f"def unsupported({signature}): pass", namespace)
    with pytest.raises(TypeError, match="keyword-compatible"):
        OpenAIAdapter()._callable_to_json_schema(namespace["unsupported"])


def test_unresolved_annotations_fail_instead_of_becoming_strings():
    def unresolved(value: MissingType):
        pass

    with pytest.raises(TypeError, match="Cannot generate JSON Schema"):
        OpenAIAdapter()._callable_to_json_schema(unresolved)


@pytest.mark.parametrize("adapter_class", [OpenAIAdapter, AnthropicAdapter,
                                          MistralAdapter, KimiAdapter, ZaiAdapter])
@pytest.mark.parametrize("function", [inspect_employee, EmployeeTool()])
def test_provider_envelopes_preserve_shared_schema(adapter_class, function):
    tool = adapter_class()._convert_function_to_tool(function)
    declaration = tool.get("function", tool)
    schema = declaration.get("parameters", declaration.get("input_schema"))
    assert Draft202012Validator(schema).is_valid(valid_arguments())
    assert not Draft202012Validator(schema).is_valid({**valid_arguments(), "employee": {}})
    assert declaration.get("strict") is not True
    if adapter_class is OpenAIAdapter:
        assert declaration["strict"] is False


def weather(city: str, note: str | None, unit: str = "C"):
    return {"city": city, "note": note, "unit": unit}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_openai_execution_preserves_omitted_defaults_and_explicit_null(asynchronous):
    adapter = OpenAIAdapter()
    tools = [adapter._convert_function_to_tool(weather)]
    schema = tools[0]["parameters"]
    assert schema["required"] == ["city", "note"]
    assert schema["properties"]["unit"]["default"] == "C"
    assert not Draft202012Validator(schema).is_valid({"city": "Paris", "note": None, "unit": None})
    calls = [FunctionCall("fc_1", "weather", json.dumps({"city": "Paris", "note": None}), "call_1")]
    if asynchronous:
        responses = asyncio.run(adapter._execute_tool_calls_async(calls, [weather], tools))
    else:
        responses = adapter._execute_tool_calls(calls, [weather], tools)
    assert responses[0].response == {"city": "Paris", "note": None, "unit": "C"}


@pytest.mark.parametrize("schema_source", [Employee, Employee.model_json_schema(), list[Employee], {}])
def test_google_structured_output_preserves_models_raw_schemas_and_typed_containers(schema_source):
    from pydantic import TypeAdapter

    expected = (schema_source if isinstance(schema_source, dict)
                else TypeAdapter(schema_source).json_schema())
    result = GoogleAdapter()._build_structured_output({"structured_output": schema_source})
    assert result == {"type": "text", "mime_type": "application/json", "schema": expected}
    if isinstance(schema_source, dict):
        assert result["schema"] is not schema_source


@pytest.mark.parametrize("disabled", [None, False])
def test_google_structured_output_can_be_disabled(disabled):
    assert GoogleAdapter()._build_structured_output({"structured_output": disabled}) is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_google_sdk_serializes_named_response_format_and_function_schemas(monkeypatch, asynchronous):
    from google import genai

    adapter = GoogleAdapter()
    adapter.model_config = {"test": {"background_mode": False, "agent_type": None}}
    adapter._client = genai.Client(api_key="test-key")
    captured = []

    def send(**kwargs):
        request = kwargs["request"]
        captured.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": "interaction_test", "status": "completed", "steps": [], "model": "test",
        })

    async def send_async(**kwargs):
        return send(**kwargs)

    kwargs = dict(model="test", the_conversation=Conversation([Message("user", "test")]),
                  functions=[inspect_employee, EmployeeTool()],
                  additional_parameters={"structured_output": Employee})
    if asynchronous:
        monkeypatch.setattr(adapter.async_client.interactions, "do_request_async", send_async)
        asyncio.run(adapter.request_llm_async(**kwargs))
    else:
        monkeypatch.setattr(adapter.client.interactions, "do_request", send)
        adapter.request_llm(**kwargs)
    assert len(captured) == 1
    body = captured[0]
    assert body["response_format"] == {
        "type": "text", "mime_type": "application/json", "schema": Employee.model_json_schema(),
    }
    assert len(body["tools"]) == 2
    for tool in body["tools"]:
        Draft202012Validator.check_schema(tool["parameters"])
        assert Draft202012Validator(tool["parameters"]).is_valid(valid_arguments())


def test_google_rejects_non_callable_tools():
    with pytest.raises(TypeError, match="callable"):
        GoogleAdapter()._build_tools([object()], {})


@pytest.mark.parametrize("function", [inspect_employee, EmployeeTool()])
def test_grok_sdk_keeps_schema_semantics(function):
    tool = GrokAdapter()._convert_function_to_tool(function)
    schema = json.loads(tool.function.parameters)
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(valid_arguments())
    assert not Draft202012Validator(schema).is_valid({**valid_arguments(), "employee": {}})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("arguments, expected", [({}, "default"), ({"value": None}, None)])
def test_callable_instances_keep_nullable_defaults_and_google_lookup(asynchronous, arguments, expected):
    class NullableTool:
        def __call__(self, value: str | None = "default"):
            return {"value": value}

    function = NullableTool()
    adapter = GoogleAdapter()
    declaration = adapter._build_tools([function], {})[0]
    assert declaration["name"] == "NullableTool"
    assert Draft202012Validator(declaration["parameters"]).is_valid(arguments)
    call = FunctionCall("call_1", declaration["name"], json.dumps(arguments))
    if asynchronous:
        responses = asyncio.run(adapter._execute_function_calls_async([call], [function], None))
    else:
        responses = adapter._execute_function_calls([call], [function], None)
    assert responses[0].response == {"value": expected}


def test_bound_methods_and_partials_use_resolved_signatures():
    class Lookup:
        def lookup(self, city: str, unit: str = "C"):
            pass

    adapter = OpenAIAdapter()
    bound = Lookup().lookup
    for function in (bound, functools.partial(bound, unit="F")):
        tool = adapter._convert_function_to_tool(function)
        assert tool["name"] == "lookup"
        assert tool["parameters"]["required"] == ["city"]
        assert set(tool["parameters"]["properties"]) == {"city", "unit"}
    assert tool["parameters"]["properties"]["unit"]["default"] == "F"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_openai_sdk_parse_accepts_non_strict_tools_with_structured_output(asynchronous):
    from openai import AsyncOpenAI, OpenAI

    captured = []

    def send(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "test", "status": "completed", "output": [],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        })

    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    kwargs = dict(model="test", the_conversation=Conversation([Message("user", "test")]),
                  functions=[weather], additional_parameters={"structured_output": Employee})
    if asynchronous:
        async def run():
            async with AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(send),
            )) as client:
                adapter._async_client = client
                await adapter.request_llm_async(**kwargs)
        asyncio.run(run())
    else:
        with OpenAI(api_key="test-key", http_client=httpx.Client(
            transport=httpx.MockTransport(send),
        )) as client:
            adapter._client = client
            adapter.request_llm(**kwargs)
    assert len(captured) == 1
    assert captured[0]["tools"][0]["strict"] is False
    assert captured[0]["tools"][0]["parameters"]["required"] == ["city", "note"]
    assert captured[0]["text"]["format"]["type"] == "json_schema"
