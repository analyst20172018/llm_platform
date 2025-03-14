from .adapter_base import AdapterBase
from openai import OpenAI
import os
from typing import List, Tuple, Callable, Dict
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
import logging

class DeepSeekAdapter(AdapterBase):
    
    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)   
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
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

                        # Check if the model supports images
                        if model == 'o1-mini':
                            logging.warning("The model 'o1-mini' does not support images yet. Image is ignored.")
                            continue

                        # Ensure that history_message["content"] is a list, not a string
                        if not isinstance(history_message["content"], list):
                            history_message["content"] = [history_message["content"]]

                        # Add the image to the content list
                        image_content = {"type": "image_url", 
                                         "image_url": {"url": f"data:image/{each_file.extension};base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    if isinstance(each_file, AudioFile):
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
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile, PDFDocumentFile)):
                        
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
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

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
                    additional_parameters: Dict={},
                    **kwargs) -> Message:
        
        if additional_parameters:
            logging.warning("Additional parameters is not supported by DeepSeek API")

        # Remove 'max_tokens' from kwargs if it exists
        max_tokens = kwargs.pop('max_tokens', None)
        if max_tokens:
            logging.warning("Max tokens parameter is removed.")

        if model != 'deepseek-reasoner':
            # Add temperature to kwargs, note that the model deepseek-reasoner does not support temperature
            kwargs['temperature'] = temperature

        history, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model, **kwargs)
        response = self.client.chat.completions.create(
                        model=model,
                        messages=history,
                        **kwargs,
                        )
        
        usage = {"model": model,
                 "completion_tokens": response.usage.completion_tokens,
                 "prompt_tokens": response.usage.prompt_tokens}
        
        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        
        return message

    def voice_to_text(self, audio_file):
        raise NotImplementedError("OpenRoute does not support voice to text")

    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        NotImplementedError("Not implemented yet")

    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[Callable], 
                                   temperature: int=0,  
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs):
        raise NotImplementedError("Not implemented yet")

    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")

    