import yaml
from pathlib import Path
from typing import List, Dict

class Model:

    def __init__(self, model_config_data: Dict) -> None:
        self.model_config_data = model_config_data

    def __getitem__(self, parameter:str) -> Dict:
        return self.model_config_data.get(parameter, None)
    
    @property
    def inputs(self) -> List:
        return self.model_config_data.get('inputs', [])
    
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
    
    # Legacy code

    def model(self, model_name) -> Dict:
        """Get the model configuration for the given model"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                return each_model
        return None
    
    def model_has_parameter(self, model_name, parameter_name) -> bool:
        """Check if the model has the given parameter"""
        for each_model in self.model_config['models']:
            if each_model['name'] == model_name:
                if parameter_name in each_model:
                    return True
        return False

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
    
    