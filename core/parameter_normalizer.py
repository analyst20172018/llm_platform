from typing import Any, Dict

from loguru import logger

from llm_platform.types import AdditionalParameters


class ParameterNormalizer:
    """Normalizes per-call ``additional_parameters`` against a model's YAML schema.

    The pipeline merges deprecated ``**kwargs``, applies ``send_default`` defaults,
    maps friendly keys to nested provider ``request_key`` paths, drops keys flagged
    ``include_in_request: false``, and filters out keys the model does not support.
    """

    def __init__(self, model_config):
        self.model_config = model_config

    def normalize(
        self,
        model: str,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return a cleaned request-parameter dict for the selected model."""
        merged = self._merge_additional_parameters(additional_parameters, kwargs)

        model_object = self.model_config[model] if model else None
        if not model_object:
            return merged

        self._apply_parameter_defaults(merged, model_object)
        self._apply_request_key_mappings(merged, model_object)
        self._remove_excluded_parameters(merged, model_object)
        return self._filter_allowed_parameters(model, merged, model_object)

    @staticmethod
    def _set_nested_parameter(target: Dict[str, Any], path: str, value: Any) -> None:
        if not path:
            return
        parts = [part for part in str(path).split(".") if part]
        if not parts:
            return
        cursor = target
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    def _allowed_parameter_keys(self, model_object: Any) -> set:
        if not model_object:
            return set()
        allowed = set()
        for name, definition in model_object.parameter_definitions().items():
            if definition.get("include_in_request", True):
                allowed.add(name)
            request_key = definition.get("request_key")
            if request_key:
                allowed.add(str(request_key).split(".")[0])
        allowed.update({"response_modalities"})
        return allowed

    @staticmethod
    def _request_root_key(request_key: str | None) -> str | None:
        if not request_key:
            return None
        return str(request_key).split(".")[0]

    @staticmethod
    def _merge_additional_parameters(
        additional_parameters: AdditionalParameters | None,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if additional_parameters:
            merged.update(additional_parameters)

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                merged.setdefault(key, value)

        return merged

    def _apply_parameter_defaults(self, merged: Dict[str, Any], model_object: Any) -> None:
        for name, definition in model_object.parameter_definitions().items():
            if not definition.get("send_default", True):
                continue
            if name in merged:
                continue

            root_key = self._request_root_key(definition.get("request_key"))
            if root_key and root_key in merged:
                continue

            default_value = definition.get("default")
            if default_value is not None:
                merged[name] = default_value

    def _apply_request_key_mappings(self, merged: Dict[str, Any], model_object: Any) -> None:
        for name, definition in model_object.parameter_definitions().items():
            request_key = definition.get("request_key")
            if not request_key or name not in merged:
                continue

            value = merged.pop(name)
            if value is None or value == "":
                continue

            self._set_nested_parameter(merged, request_key, value)

    @staticmethod
    def _remove_excluded_parameters(merged: Dict[str, Any], model_object: Any) -> None:
        for name, definition in model_object.parameter_definitions().items():
            if not definition.get("include_in_request", True):
                merged.pop(name, None)

    def _filter_allowed_parameters(
        self,
        model: str,
        merged: Dict[str, Any],
        model_object: Any,
    ) -> Dict[str, Any]:
        allowed = self._allowed_parameter_keys(model_object)
        if not allowed:
            return merged

        filtered: Dict[str, Any] = {}
        for key, value in merged.items():
            if key in allowed:
                filtered[key] = value
            else:
                logger.warning(
                    f"Model {model} does not support parameter '{key}'. Ignoring the parameter."
                )
        return filtered
