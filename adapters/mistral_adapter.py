from .adapter_base import AdapterBase
from mistralai import Mistral
import os
from typing import List, Tuple, Callable, Dict
import logging
from llm_platform.tools.base import BaseTool
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
import json
import inspect

class MistralAdapter(AdapterBase):
    
    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)   
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def convert_conversation_history_to_adapter_format(self, 
                        the_conversation: Conversation, 
                        model: str, 
                        **kwargs):
        
        # Add system prompt as the message from the user "system"
        history = [{"role": "system", "content": the_conversation.system_prompt}]

        # Add history of messages
        for message in the_conversation.messages:

            history_message = {"role": message.role, "content": message.content}

            if not message.files is None:
                # Ensure that history_message["content"] is a list, not a string
                if not isinstance(history_message["content"], list):
                    history_message["content"] = [{
                        "type": "text",
                        "text": history_message["content"]
                    }]

                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Add the image to the content list
                        image_content = {"type": "image_url", 
                                         "image_url": {"url": f"data:image/{each_file.extension};base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(image_content)
                    
                    # Audio
                    if isinstance(each_file, AudioFile):
                        logging.warning("Audio files are not supported by Mistral API and it is skipped")
                    
                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile)):

                        # Add the text document to the history as a text in XML tags
                        new_text_content = {
                            "type": "text",
                            "text": f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        }
                        history_message["content"].insert(0, new_text_content)

                    # PDF documents
                    elif isinstance(each_file, PDFDocumentFile):
                        
                        # Add the image to the content list
                        pdf_content = {"type": "document_url", 
                                         "document_url": {"url": f"data:application/pdf;base64,{each_file.base64}"}
                                         }
                        history_message["content"].append(pdf_content)

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            history.append(history_message)

            # Add all function responses to the history
            if message.function_responses:
                for each_response in message.function_responses:
                    history.append(each_response.to_openai())

        return history, kwargs

    def convert_conversation_history_to_adapter_format_for_ocr(self, the_conversation: Conversation) -> Dict:
        
        # Add history of messages
        for message in the_conversation.messages:

            if message.files:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):

                        # Add the image to the content list
                        document = {"type": "image_url", 
                                    "image_url": f"data:image/{each_file.extension};base64,{each_file.base64}"
                        }
                        return document
                    
                    # Audio
                    if isinstance(each_file, AudioFile):
                        logging.warning("Audio files are not supported")
                    
                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile)):
                        logging.warning("Text files are not supported")

                    # PDF documents
                    elif isinstance(each_file, PDFDocumentFile):
                        # Add the image to the content list
                        document = {
                            "type": "document_url", 
                            "document_url": f"data:application/pdf;base64,{each_file.base64}" 
                        }
                        return document

                    else:
                        raise ValueError(f"Unsupported file type: {message.file.get_type()}")

            raise ValueError(f"No image or pdf files are found in the conversation history")

        return None

    def request_llm(self, model: str, 
                    the_conversation: Conversation, 
                    functions:List[BaseTool]=None, 
                    temperature: int=0, 
                    tool_output_callback: Callable=None, 
                    additional_parameters: Dict={},
                    **kwargs):

        if additional_parameters:
            logging.warning("Additional parameters is not supported by Mistral API")

        kwargs['temperature'] = temperature

        # OCR
        if model == "mistral-ocr-latest":
            document = self.convert_conversation_history_to_adapter_format_for_ocr(the_conversation)
            ocr_response = self.client.ocr.process(
                                        model=model,
                                        document=document,
                                        include_image_base64=False
                            )
            response_pages_markdown = [each.markdown for each in ocr_response.pages]
            output_markdown = '\n'.join(response_pages_markdown)

            message = Message(role="assistant", content=output_markdown)
            the_conversation.messages.append(message)

            return output_markdown
        
        # LLM with functions
        elif not functions is None:
            response = self.request_llm_with_functions(
                            model=model,
                            the_conversation=the_conversation,
                            functions=functions,
                            tool_output_callback=tool_output_callback,
                            **kwargs,
                        )
            
        # Standard text LLM
        else:
            messages, kwargs = self.convert_conversation_history_to_adapter_format(the_conversation, model, **kwargs)
            response = self.client.chat.complete(
                            model=model,
                            messages=messages,
                            **kwargs,
                            )
        
        usage = {"model": model,
                 "completion_tokens": 0, #response.usage.completion_tokens,
                 "prompt_tokens": 0 #response.usage.prompt_tokens
                }
        
        message = Message(role="assistant", content=response.choices[0].message.content, usage=usage)
        the_conversation.messages.append(message)
        
        return response.choices[0].message.content
    
    def request_llm_with_functions(self, model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool | Callable], 
                                   tool_output_callback: Callable=None,
                                   **kwargs):
        raise NotImplementedError("Not implemented yet")
    
    def voice_to_text(self, audio_file):
        raise NotImplementedError("Mistral does not support voice to text")

    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        raise NotImplementedError("Not implemented yet")

    def get_models(self) -> List[str]:
        raise NotImplementedError("Not implemented yet")