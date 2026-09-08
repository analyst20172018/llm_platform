"""JSON output formats shared by Chat Completions providers."""

import copy
import json


def response_format(value):
    """Accept JSON mode, a Pydantic class, a schema, or a native envelope."""
    if value is True:
        return {"type": "json_object"}
    if hasattr(value, "model_json_schema"):
        schema = value.model_json_schema()
        name = value.__name__
    elif isinstance(value, dict):
        if value.get("type") == "json_object":
            return {"type": "json_object"}
        if value.get("type") == "json_schema":
            envelope = value.get("json_schema")
            if not isinstance(envelope, dict) or not isinstance(envelope.get("schema"), dict):
                raise TypeError("structured_output json_schema must contain a schema dict")
            return copy.deepcopy(value)
        schema = copy.deepcopy(value)
        name = str(schema.get("title", "response"))
    else:
        raise TypeError("structured_output must be True, a Pydantic model class, or a JSON Schema dict")
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def add_json_instruction(history, output_format):
    """Add JSON guidance to detached wire history, preserving conversation state."""
    instruction = 'Return only a valid JSON object (for example, {"answer": "..."}).'
    if output_format["type"] == "json_schema":
        instruction += " Match this JSON Schema:\n" + json.dumps(
            output_format["json_schema"]["schema"], ensure_ascii=False
        )
    system = history[0].get("content") or ""
    history[0]["content"] = f"{system}\n\n{instruction}" if system else instruction
