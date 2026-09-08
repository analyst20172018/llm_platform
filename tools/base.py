import copy
import shlex
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Literal

from pydantic import BaseModel


class BaseTool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        raise NotImplementedError

    # Shell control operators that allow a single command string to invoke more than
    # one program (chaining, piping, substitution, redirection). When an allow-list is
    # active these are rejected, so checking only the leading token is a sound guard.
    _SHELL_CONTROL_CHARS = (";", "|", "&", "`", "$", ">", "<", "(", ")", "\n", "\r")

    def _check_command_allowed(self, command: str, allowed_commands: Iterable[str] | None) -> None:
        """Opt-in command allow-list guard for command-executing tools.

        When ``allowed_commands`` is None (the default) no restriction is applied, so
        existing behavior is unchanged. When a collection is supplied, the command must
        be a single simple command (no shell chaining/piping/substitution/redirection)
        whose leading token is one of the allowed names; otherwise a ``PermissionError``
        is raised before the command runs. The shell-operator check is what makes the
        leading-token allow-list a real boundary rather than a bypassable hint.
        """
        if allowed_commands is None:
            return
        if any(char in command for char in self._SHELL_CONTROL_CHARS):
            raise PermissionError(
                f"Command contains a disallowed shell control operator; only a single "
                f"simple command from the allow-list is permitted for {self.name}."
            )
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        first_token = tokens[0] if tokens else ""
        if first_token not in set(allowed_commands):
            raise PermissionError(
                f"Command {first_token!r} is not in the allow-list for {self.name}."
            )

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def clean_schema(cls, data: Any) -> Any:
        """Copy a schema, removing titles and legacy ``Field(required=True)`` extras.

        Only visit schema positions: property/definition names and literal data
        in defaults, enums, and examples may themselves be schema keywords.
        Keep real required lists, references, and all validation constraints.
        """
        if not isinstance(data, dict):
            return copy.deepcopy(data)

        cleaned = copy.deepcopy(data)
        cleaned.pop("title", None)
        if isinstance(cleaned.get("required"), bool):
            cleaned.pop("required")

        for key in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            if isinstance(cleaned.get(key), dict):
                cleaned[key] = {
                    name: cls.clean_schema(schema)
                    for name, schema in cleaned[key].items()
                }
        for key in (
            "items", "additionalProperties", "additionalItems", "contains",
            "propertyNames", "not", "if", "then", "else",
            "unevaluatedProperties", "unevaluatedItems",
        ):
            if isinstance(cleaned.get(key), dict):
                cleaned[key] = cls.clean_schema(cleaned[key])
        for key in ("anyOf", "oneOf", "allOf", "prefixItems", "items"):
            if isinstance(cleaned.get(key), list):
                cleaned[key] = [cls.clean_schema(schema) for schema in cleaned[key]]
        return cleaned

    @classmethod
    def resolve_schema_for_google(cls, schema: Dict) -> Dict:
        """Compatibility helper: Interactions accepts JSON Schema references.

        Preserve references rather than expanding them, which both loses sibling
        constraints and fails on recursive definitions. Endpoint-specific schema
        restrictions must not be implemented by silently discarding constraints.
        """
        return copy.deepcopy(schema)

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
                "parameters": cleaned_schema,
            }

        if provider == "anthropic":
            return {
                "name": cls.__name__,
                "description": cls.__doc__,
                "input_schema": cleaned_schema,
            }

        if provider == "grok":
            # Imported lazily so the tool layer does not hard-depend on the xai_sdk.
            from xai_sdk.chat import tool

            return tool(
                name=cls.__name__,
                description=cls.__doc__,
                parameters=cleaned_schema,
            )

        raise NotImplementedError
