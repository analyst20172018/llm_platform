from .adapter_base import AdapterBase
from openai import OpenAI
import os
from typing import Any, Callable, Dict, List, Tuple
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile)
from loguru import logger
from llm_platform.types import AdditionalParameters

class OpenRouterAdapter(AdapterBase):
    
    def __init__(self):
        super().__init__()   
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    def convert_conversation_history_to_adapter_format(self, 
                        the_conversation: Conversation, 
                        model: str, 
                        **kwargs):
        
        # Add system prompt as the message from the user "system"
        history = [{"role": "system", "content": the_conversation.system_prompt}]

        # Add history of messages
        for message in the_conversation.messages:

            history_message = {"role": message.role, "content": message.content}

            # If there is an attribute tool_calls in message, then add it to history. 
            if message.function_calls:
                history_message["tool_calls"] = [each_call.to_openai() for each_call in message.function_calls]

            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]

                        # Add the image to the content list
                        image_content = {"type": "image_url", 
                                         "image_url": {"url": f"data:image/{each_file.extension};base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    elif isinstance(each_file, AudioFile):
                        # Update kwargs as needed
                        assert "gpt-4o-audio" in model
                        if 'modalities' not in kwargs:
                            kwargs['modalities'] = ["text"] #["text", "audio"]
                        if 'audio' not in kwargs:
                            kwargs['audio'] = {"voice": "alloy", "format": "wav"}

                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]

                        # Add audio to history
                        audio_content = {
                            "type": "input_audio",
                            "input_audio": {
                                "data": each_file.base64,
                                "format": "mp3"
                            }
                        }
                        history_message["content"].append(audio_content)
                    
                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile)):
                        
                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [{
                                "type": "text",
                                "text": history_message["content"]
                            }]

                        # Add the text document to the history as a text in XML tags
                        new_text_content = {
                            "type": "text",
                            "text": f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        }
                        history_message["content"].insert(0, new_text_content)

                    else:
                        raise ValueError(f"Unsupported file type in file {each_file.name}. The type is {type(each_file)}.")

            history.append(history_message)

            # Add all function responses to the history
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(each_response.to_openai())

        return history, kwargs

    def request_llm(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: AdditionalParameters | None = None,
                    **kwargs) -> Message:

        if additional_parameters is None:
            additional_parameters = {}

        if temperature not in (None, 0) and "temperature" not in additional_parameters:
            additional_parameters["temperature"] = temperature

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)

        history, history_kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model)

        request_params: Dict[str, Any] = {}
        if "temperature" in additional_parameters:
            request_params["temperature"] = additional_parameters["temperature"]
        if "max_tokens" in additional_parameters:
            request_params["max_tokens"] = additional_parameters["max_tokens"]

        reserved = {
            "response_modalities",
            "web_search",
            "code_execution",
            "citations_enabled",
            "url_context",
            "structured_output",
            "reasoning",
            "text",
            "temperature",
            "max_tokens",
        }
        for key, value in additional_parameters.items():
            if key in reserved:
                continue
            request_params[key] = value

        request_params.update(history_kwargs)

        response = self.client.chat.completions.create(
                        model=model,
                        messages=history,
                        **request_params,
                        )
        if getattr(response, 'usage', None):
            if hasattr(response.usage, 'completion_tokens'):
                completion_tokens = response.usage.completion_tokens
            else:
                completion_tokens = 0

            if hasattr(response.usage, 'prompt_tokens'):
                prompt_tokens = response.usage.prompt_tokens
            else:
                prompt_tokens = 0

        usage = {"model": model,
                 "completion_tokens": completion_tokens,
                 "prompt_tokens": prompt_tokens}
        
        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        
        return message

    def voice_to_text(self, audio_file):
        raise NotImplementedError("OpenRoute does not support voice to text")

    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        raise NotImplementedError("Not implemented yet")

    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[Callable], 
                                   temperature: int=0,  
                                   tool_output_callback: Callable=None,
                                   additional_parameters: AdditionalParameters | None = None,
                                   **kwargs):
        raise NotImplementedError("Not implemented yet")

    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")

    
