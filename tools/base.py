import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

from pydantic import BaseModel
from xai_sdk.chat import tool


class BaseTool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        raise NotImplementedError

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def clean_schema(cls, data: Any) -> Any:
        """Recursively remove unsupported metadata from provider schemas."""
        if isinstance(data, list):
            return [cls.clean_schema(item) for item in data]

        if not isinstance(data, dict):
            return data

        cleaned = dict(data)
        cleaned.pop("title", None)

        properties = cleaned.get("properties")
        if isinstance(properties, dict):
            cleaned_properties = {}
            for property_name, property_data in properties.items():
                if isinstance(property_data, dict):
                    property_data = dict(property_data)
                    property_data.pop("required", None)
                    property_data.pop("title", None)
                cleaned_properties[property_name] = cls.clean_schema(property_data)
            cleaned["properties"] = cleaned_properties

            if isinstance(cleaned.get("required"), list):
                property_names = set(cleaned_properties.keys())
                cleaned["required"] = [
                    name for name in cleaned["required"] if name in property_names
                ]

        for key, value in list(cleaned.items()):
            if key == "properties":
                continue
            if isinstance(value, (dict, list)):
                cleaned[key] = cls.clean_schema(value)

        return cleaned

    @classmethod
    def resolve_schema_for_google(cls, schema: Dict) -> Dict:
        """
        Make a JSON Schema compatible with Google Gemini's function declaration validator.

        Gemini does not support $ref/$defs, anyOf, exclusiveMinimum,
        exclusiveMaximum, or additionalProperties.
        """
        schema = copy.deepcopy(schema)
        definitions = schema.pop("$defs", {})

        def resolve(node: Any) -> Any:
            if isinstance(node, list):
                return [resolve(item) for item in node]
            if not isinstance(node, dict):
                return node

            if "$ref" in node:
                reference_key = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(definitions.get(reference_key, {})))

            if "anyOf" in node:
                non_null_variants = [
                    item for item in node["anyOf"] if item != {"type": "null"}
                ]
                if len(non_null_variants) == 1:
                    resolved = dict(resolve(non_null_variants[0]))
                    for key in ("description", "default"):
                        if key in node and key not in resolved:
                            resolved[key] = node[key]
                    return resolved

            result = {}
            for key, value in node.items():
                if key in (
                    "$defs",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "additionalProperties",
                ):
                    continue
                result[key] = resolve(value)
            return result

        return resolve(schema)

    @classmethod
    def to_params(cls, provider: Literal["anthropic", "openai", "google", "grok"]) -> Dict:
        input_model = cls.InputModel
        if not issubclass(input_model, BaseModel):
            raise ValueError("InputModel must be a Pydantic BaseModel")

        schema = input_model.model_json_schema()
        cleaned_schema = cls.clean_schema(schema)

        if provider == "openai":
            return {
                "name": cls.__name__,
                "description": cls.__doc__,
                "parameters": cleaned_schema,
            }

        if provider == "google":
            return {
                "name": cls.__name__,
                "description": cls.__doc__,
                "parameters": cls.clean_schema(cls.resolve_schema_for_google(schema)),
            }

        if provider == "anthropic":
            return {
                "name": cls.__name__,
                "description": cls.__doc__,
                "input_schema": cleaned_schema,
            }

        if provider == "grok":
            return tool(
                name=cls.__name__,
                description=cls.__doc__,
                parameters=cleaned_schema,
            )

        raise NotImplementedError
