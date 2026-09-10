"""Offline checks for lossless provider response serialization."""

from copy import deepcopy
import json
import warnings

import httpx
from openai import OpenAI
from pydantic import BaseModel
import pytest

from llm_platform.adapters.serializers import provider_dump


class Vacancy(BaseModel):
    title: str


@pytest.mark.parametrize("output_only", [False, True])
def test_provider_dump_preserves_openai_structured_response_without_warnings(output_only):
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "test",
        "status": "completed",
        "output": [{
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": '{"title":"Engineer"}',
                "annotations": [],
            }],
        }],
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client:
        response = client.responses.parse(model="test", input="Find a vacancy", text_format=Vacancy)

    assert response.output_parsed == Vacancy(title="Engineer")
    expected = deepcopy(payload)
    expected["output"][0]["content"][0]["parsed"] = {"title": "Engineer"}
    value = response.output if output_only else response
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumped = provider_dump(value)

    assert dumped == (expected["output"] if output_only else expected)
    assert json.loads(json.dumps(dumped)) == dumped
