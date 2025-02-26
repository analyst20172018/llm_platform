from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import List, Tuple, Callable, Dict
import logging
from llm_platform.services.conversation import Conversation
from llm_platform.tools.base import BaseTool

class AdapterBase(ABC):
    def __init__(self, logging_level=logging.INFO):
        load_dotenv()
        self.latest_usage = None
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    @abstractmethod
    def convert_conversation_history_to_adapter_format(self, the_conversation: Conversation):
        pass

    @abstractmethod
    def request_llm(self, model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    temperature: int=0,
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={}, 
                    **kwargs):
        pass

    @abstractmethod
    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool], 
                                   temperature: int=0,
                                   tool_output_callback: Callable=None,
                                   **kwargs):
        pass

    @abstractmethod
    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        pass

    """
    # --- Asynchronous Methods ---
    @abstractmethod
    async def request_llm_async(self, model: str,
                        the_conversation: Conversation,
                        functions:List[BaseTool]=None,
                        temperature: int=0,
                        tool_output_callback: Callable=None,
                        additional_parameters: Dict={},
                        **kwargs):
        pass

    @abstractmethod
    async def request_llm_with_functions_async(self, model: str,
                                       the_conversation: Conversation,
                                       functions: List[BaseTool],
                                       temperature: int=0,
                                       tool_output_callback: Callable=None,
                                       **kwargs):
        pass

    @abstractmethod
    async def generate_image_async(self, prompt: str, size: str, quality:str, n=1):
        pass
    """
