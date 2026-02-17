import copy
from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Dict, Any
from pydantic import BaseModel, Field
import subprocess
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
    def _convert_type_names(cls, type_name: str) -> str:
        type_names_convertor = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        return type_names_convertor.get(type_name, type_name)

    @classmethod
    def clean_schema(cls, data: Any) -> Any:
        """
        Recursively remove any "title" keys and any "required": true
        entries from the schema.
        """
        if isinstance(data, dict):
            # Remove the "title" key if it exists
            data.pop("title", None)

            # Remove the "required": true from individual properties
            # (but do NOT remove the top-level "required" array if present)
            if "properties" in data:
                for prop_name, prop_data in data["properties"].items():
                    prop_data.pop("required", None)  # removes "required": true
                    prop_data.pop("title", None)     # removes any "title" in each property
                    # Recursively clean nested dictionaries or lists under each property
                    cls.clean_schema(prop_data)

            # Recursively clean other nested objects
            for key, value in list(data.items()):
                if isinstance(value, (dict, list)):
                    data[key] = cls.clean_schema(value)

        elif isinstance(data, list):
            # Clean each item in the list
            for i in range(len(data)):
                data[i] = cls.clean_schema(data[i])

        return data

    @classmethod
    def resolve_schema_for_google(cls, schema: Dict) -> Dict:
        """
        Make a JSON Schema compatible with Google Gemini's function declaration validator.

        Gemini does not support $ref/$defs, anyOf, exclusiveMinimum,
        exclusiveMaximum, or additionalProperties.
        This method:
          - Inlines all $ref references (recursively)
          - Drops the $defs section
          - Flattens anyOf: [T, {type: null}] to just T (preserving description/default)
          - Removes exclusiveMinimum / exclusiveMaximum / additionalProperties keywords
        """
        schema = copy.deepcopy(schema)
        defs = schema.pop("$defs", {})

        def resolve(node: Any) -> Any:
            if isinstance(node, list):
                return [resolve(item) for item in node]
            if not isinstance(node, dict):
                return node

            # Inline $ref
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs.get(ref_key, {})))

            # Flatten Optional[T]: anyOf: [T, {type: "null"}]
            if "anyOf" in node:
                non_null = [x for x in node["anyOf"] if x != {"type": "null"}]
                if len(non_null) == 1:
                    resolved = resolve(non_null[0])
                    resolved = dict(resolved)
                    for key in ("description", "default"):
                        if key in node and key not in resolved:
                            resolved[key] = node[key]
                    return resolved

            # Recurse, dropping unsupported keywords
            result = {}
            for k, v in node.items():
                if k in ("$defs", "exclusiveMinimum", "exclusiveMaximum",
                         "additionalProperties"):
                    continue
                result[k] = resolve(v)
            return result

        return resolve(schema)

    @classmethod
    def to_params(cls, provider: Literal["anthropic", "openai", "google", "grok"]) -> Dict:
        input_model = cls.InputModel
        if not issubclass(input_model, BaseModel):
            raise ValueError("InputModel must be a Pydantic BaseModel")
        schema = input_model.model_json_schema()
        if provider == 'openai':
            return {
                'name': cls.__name__,
                'description': cls.__doc__,
                'parameters': cls.clean_schema(schema)
            }
        elif provider == 'google':
            return {
                "name": cls.__name__,
                "description": cls.__doc__,
                "parameters": cls.clean_schema(cls.resolve_schema_for_google(schema)),
            }
        elif provider == 'anthropic':
            return {
                'name': cls.__name__,
                'description': cls.__doc__,
                'input_schema': cls.clean_schema(schema)
            }
        elif provider == 'grok':
            return tool(
                name = cls.__name__,
                description = cls.__doc__,
                parameters=cls.clean_schema(schema),
            )
        else:
            raise NotImplementedError
        