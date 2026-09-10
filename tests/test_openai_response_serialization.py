"""Offline regressions for parsed OpenAI responses and shared serialization."""

import asyncio
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import warnings

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, field_serializer, model_serializer
import pytest

from llm_platform.adapters import serializers
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.conversation import Conversation


class Vacancy(BaseModel):
    isco_group_number: int
    isco_group_title: str


EXPECTED = {"isco_group_number": 2, "isco_group_title": "Professionals"}
REASONING = {
    "id": "rs_test", "type": "reasoning", "encrypted_content": "opaque-test",
    "summary": [{"type": "summary_text", "text": "Classified the occupation."}],
}
MESSAGE = {
    "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
    "content": [{
        "type": "output_text", "text": json.dumps(EXPECTED), "logprobs": [],
        "annotations": [{"type": "url_citation", "start_index": 0, "end_index": 1,
                         "title": "Example", "url": "https://example.com"}],
    }],
}


def payload(outputs):
    return {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": "gpt-5.6-luna", "status": "completed", "error": None,
        "output": deepcopy(outputs),
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 2}},
    }


@pytest.mark.parametrize("parsed,outputs", [
    (True, [REASONING, MESSAGE]),
    (False, [REASONING, MESSAGE]),
    (True, [{**MESSAGE, "content": [{"type": "refusal", "refusal": "Cannot comply."}]}]),
    (True, [{"type": "function_call", "id": "fc_test", "call_id": "call_test",
             "name": "lookup", "arguments": "{}", "status": "completed"}]),
])
def test_response_and_replay(parsed, outputs):
    body = payload(outputs)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client:
            if parsed:
                response = client.responses.parse(model="gpt-5.6-luna", input="test", text_format=Vacancy)
            else:
                response = client.responses.create(model="gpt-5.6-luna", input="test")
        dumped = serializers.provider_dump(response)
        expected = deepcopy(body)
        if parsed:
            for output in expected["output"]:
                if output["type"] == "message":
                    for content in output["content"]:
                        if content["type"] == "output_text":
                            content["parsed"] = EXPECTED
                elif output["type"] == "function_call":
                    output["parsed_arguments"] = None
        assert dumped == expected
        assert dumped["output"] == serializers.provider_dump(response.output)
        assert dumped["usage"] == body["usage"]
        assert serializers.openai_output_to_input(response.output) == outputs
        assert json.loads(json.dumps(dumped)) == dumped
        if parsed and outputs == [REASONING, MESSAGE]:
            assert dumped["output"][1]["content"][0]["parsed"] == EXPECTED
            assert response.output_parsed.model_dump() == EXPECTED


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_handler(asynchronous):
    body = payload([REASONING, MESSAGE])
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    handler = APIHandler()
    adapter = handler.get_adapter("gpt-5.6-luna")
    params = {"reasoning": {"effort": "low"}, "structured_output": Vacancy}

    async def run_async():
        async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport)) as client:
            adapter._async_client = client
            return await handler.request_async("gpt-5.6-luna", "Synthetic vacancy", additional_parameters=params)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if asynchronous:
            result = asyncio.run(run_async())
        else:
            with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client:
                adapter._client = client
                result = handler.request("gpt-5.6-luna", "Synthetic vacancy", additional_parameters=params)
    assert json.loads(result.content) == EXPECTED
    assert result.thinking_responses[0].content.strip() == "Classified the occupation."
    restored = Conversation.read_from_json(json.loads(json.dumps(handler.the_conversation.save_to_json())))
    assert restored.messages[-1].provider_data == result.provider_data
    assert restored.messages[-1].thinking_responses[0].content == result.thinking_responses[0].content


def test_normal_pydantic_semantics():
    class Record(BaseModel):
        value: int = Field(serialization_alias="wire_value")
        day: date
        optional: str | None = None
        unset: int = 7
        hidden: str = Field(exclude=True)

        @field_serializer("value")
        def serialize_value(self, value):
            return str(value)

    record = Record(value=3, day=date(2026, 9, 10), optional=None, hidden="private")
    assert serializers.provider_dump(record) == {
        "wire_value": "3", "day": "2026-09-10", "optional": None,
    }


def test_lazy_nested_sdk_model():
    # model_construct reproduces SDK construction without eagerly validating
    # and building the nested Summary serializer.
    from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary

    item = ResponseReasoningItem.model_construct(
        id="rs_test", type="reasoning",
        summary=[Summary.model_construct(type="summary_text", text="Summary")],
    )
    assert serializers.provider_dump(item)["summary"] == [{"type": "summary_text", "text": "Summary"}]


def test_reasoning_serialization_in_fresh_process():
    # Run the complete parsed response before any other test can build nested
    # SDK serializers. run_path loads definitions without running the tests.
    script = """
import runpy
import sys
sys.path.insert(0, sys.argv[1])
tests = runpy.run_path(sys.argv[2])
tests['test_response_and_replay'](True, [tests['REASONING'], tests['MESSAGE']])
tests['test_lazy_nested_sdk_model']()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2]), __file__],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_serializer_does_not_import_openai():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from pydantic import BaseModel
from llm_platform.adapters.serializers import provider_dump
class Record(BaseModel):
    value: int
assert provider_dump(Record(value=1)) == {'value': 1}
assert 'openai' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_parsed_payload_uses_custom_serializer_and_preserves_unset_and_null():
    from openai.types.responses import ParsedResponseOutputText

    class Payload(BaseModel):
        value: int

        @model_serializer
        def serialize(self):
            return {"custom": str(self.value)}

    base = {"type": "output_text", "text": "test", "annotations": []}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fields, expected in [
            ({}, base),
            ({"parsed": None}, {**base, "parsed": None}),
            ({"parsed": Payload(value=3)}, {**base, "parsed": {"custom": "3"}}),
        ]:
            value = ParsedResponseOutputText.model_construct(**base, **fields)
            assert serializers.provider_dump(value) == expected


def test_other_models_keep_annotation_based_serialization():
    class Parent(BaseModel):
        value: int

    class Child(Parent):
        extra: str

    class Container(BaseModel):
        item: Parent

    value = Container(item=Child(value=1, extra="not part of the declared schema"))
    assert serializers.provider_dump(value) == {"item": {"value": 1}}


def test_serialization_errors_propagate():
    from pydantic_core import PydanticSerializationError
    from openai.types.responses import ParsedResponseOutputText

    class BrokenPayload(BaseModel):
        @model_serializer
        def serialize(self):
            raise ValueError("broken serializer")

    value = ParsedResponseOutputText.model_construct(parsed=BrokenPayload())
    with pytest.raises(PydanticSerializationError, match="broken serializer"):
        serializers.provider_dump(value)


def test_other_models_still_emit_serialization_warnings():
    class Record(BaseModel):
        value: int

    record = Record.model_construct(value="wrong type")
    with pytest.warns(UserWarning, match="PydanticSerializationUnexpectedValue"):
        assert serializers.provider_dump(record) == {"value": "wrong type"}
