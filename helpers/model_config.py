import yaml
from pathlib import Path
from typing import Any, Dict, List

class Model:

    @staticmethod
    def _as_ratio_string(item) -> str:
        """Convert YAML-loaded ratio values (including base-60 ints) into colon-separated strings."""
        if isinstance(item, dict) and len(item) == 1:
            key, value = next(iter(item.items()))
            return f"{key}:{value}"

        if isinstance(item, (int, float)):
            whole = int(item)
            high, low = divmod(whole, 60)
            if high == 0:
                return str(whole)
            return f"{high}:{low}"

        return str(item)

    @staticmethod
    def _as_string_list(raw_value) -> List[str]:
        if not isinstance(raw_value, list):
            return []

        return [str(item) for item in raw_value if item is not None]

    @staticmethod
    def _normalize_parameter_type(raw_type: Any) -> str:
        if raw_type is None:
            return "string"
        return str(raw_type).strip().lower()

    @staticmethod
    def _format_parameter_label(name: str) -> str:
        if not name:
            return ""
        cleaned = name.replace("_", " ").replace("-", " ").strip()
        if not cleaned:
            return name
        return cleaned[0].upper() + cleaned[1:]

    def __init__(self, model_config_data: Dict) -> None:
        self.model_config_data = model_config_data
        self._parameter_definitions = self._load_parameter_definitions()

    def _load_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        raw_parameters = self.model_config_data.get("additional_parameters")
        if not raw_parameters:
            return {}

        if isinstance(raw_parameters, list):
            raw_parameters = {
                item.get("name"): {key: value for key, value in item.items() if key != "name"}
                for item in raw_parameters
                if isinstance(item, dict) and item.get("name")
            }

        if not isinstance(raw_parameters, dict):
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        for name, definition in raw_parameters.items():
            if not name:
                continue
            if definition is None:
                definition = {}
            if not isinstance(definition, dict):
                definition = {"default": definition}

            normalized_definition = dict(definition)
            normalized_definition["name"] = name
            param_type = self._normalize_parameter_type(normalized_definition.get("type"))
            normalized_definition["type"] = param_type

            if "label" not in normalized_definition:
                normalized_definition["label"] = self._format_parameter_label(name)

            if "ui" not in normalized_definition:
                if param_type in {"boolean", "bool"}:
                    normalized_definition["ui"] = "toggle"
                elif param_type == "enum":
                    normalized_definition["ui"] = "select"
                else:
                    normalized_definition["ui"] = "input"

            if "send_default" not in normalized_definition:
                normalized_definition["send_default"] = True

            if "include_in_request" not in normalized_definition:
                normalized_definition["include_in_request"] = True

            if param_type == "enum":
                options = normalized_definition.get("options")
                if name == "aspect_ratio":
                    normalized_definition["options"] = [
                        self._as_ratio_string(item) for item in options or [] if item is not None
                    ]
                else:
                    normalized_definition["options"] = self._as_string_list(options)

            normalized[name] = normalized_definition

        return normalized

    def __getitem__(self, parameter:str) -> Dict:
        return self.model_config_data.get(parameter, None)

    @property
    def name(self) -> str:
        return self.model_config_data.get('name')

    @property
    def display_name(self) -> str:
        return self.model_config_data.get('display_name', self.name)

    @property
    def inputs(self) -> List:
        return self.model_config_data.get('inputs', [])

    @property
    def outputs(self) -> List:
        return self.model_config_data.get('outputs', [])

    @property
    def pricing(self) -> Dict:
        return self.model_config_data.get('pricing', None)
    
    @property
    def adapter(self) -> str:
        return self.model_config_data.get('adapter', None)

    @property
    def max_tokens(self) -> int:
        return self.model_config_data.get('max_tokens', None)

    @property
    def context_window(self) -> int:
        return self.model_config_data.get('context_window', None)

    @property
    def visible(self) -> bool:
        """Return whether the model should be displayed in selection lists."""
        return self.model_config_data.get('visible', True)

    @property
    def reasoning_effort(self) -> List[str]:
        definition = self.get_parameter("reasoning_effort")
        if definition:
            return definition.get("options", [])
        return []

    @property
    def verbosity(self) -> List[str]:
        definition = self.get_parameter("verbosity")
        if definition:
            return definition.get("options", [])
        return []

    @property
    def aspect_ratio(self) -> List[str]:
        definition = self.get_parameter("aspect_ratio")
        if definition:
            return definition.get("options", [])
        return []

    @property
    def resolution(self) -> List[str]:
        definition = self.get_parameter("resolution")
        if definition:
            return definition.get("options", [])
        return []

    @property
    def additional_parameters(self) -> List[Dict[str, any]]:
        return list(self._parameter_definitions.values())

    def parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._parameter_definitions)

    def get_parameter(self, name: str) -> Dict[str, Any] | None:
        return self._parameter_definitions.get(name)

    def get_parameter_options(self, name: str) -> List[str]:
        definition = self.get_parameter(name)
        if not definition:
            return []
        options = definition.get("options")
        if isinstance(options, list):
            return options
        return []

    def has_parameter(self, name: str) -> bool:
        return name in self._parameter_definitions

class ModelConfig:

    def __init__(self) -> None:
        self.model_config = self.load_model_config()

    def __getitem__(self, model_name:str) -> Model:
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return Model(each_model)
        return None

    @staticmethod
    def load_model_config():
        config_path = Path(__file__).parent.parent / 'models_config.yaml'
        with config_path.open('r', encoding="utf-8") as file:
            return yaml.safe_load(file)

    def model_names(self) -> List[str]:
        """Get the list of all the models"""
        model_names = []
        for each_model in self.model_config['models']:
            model_names.append(each_model['name'])
        return model_names

    def models(self, visible_only: bool = False) -> List[Model]:
        """Return Model wrappers for config entries, optionally filtering visibility."""
        models = [Model(each_model) for each_model in self.model_config['models']]
        if not visible_only:
            return models
        return [model for model in models if model.visible]

    def group_models_by_adapter(self, visible_only: bool = True) -> List[Dict[str, List[Model]]]:
        """Return models grouped by adapter while preserving YAML ordering."""
        groups: List[Dict[str, List[Model]]] = []
        adapter_to_group: Dict[str, Dict[str, List[Model]]] = {}

        for raw_model in self.model_config['models']:
            model = Model(raw_model)
            if visible_only and not model.visible:
                continue

            adapter_key = model.adapter or 'Unknown'

            if adapter_key not in adapter_to_group:
                group = {'adapter': adapter_key, 'models': []}
                adapter_to_group[adapter_key] = group
                groups.append(group)

            adapter_to_group[adapter_key]['models'].append(model)

        return groups
    
    # Legacy code

    def model(self, model_name) -> Dict:
        """Get the model configuration for the given model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model
        return None
    
    def model_has_parameter(self, model_name, parameter_name) -> bool:
        """Check if the model has the given parameter"""
        model = self[model_name]
        if not model:
            return False
        if model.has_parameter(parameter_name):
            return True
        return parameter_name in (model.model_config_data or {})

    def possible_inputs(self, model_name) -> List:
        """Get the possible inputs for the model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model.get('inputs', [])
        return []

    def get_pricing(self, model_name):
        """Get the pricing for the model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model.get('pricing', None)
        return None
    
    def get_adapter_name(self, model_name):
        """Get the name of the adapter for the given model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model.get('adapter', None)
        return None
    
    def get_max_tokens(self, model_name):
        """Get the max_tokens for the model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model.get('max_tokens', None)
        return None
    
    
