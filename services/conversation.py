from typing import List, Dict, BinaryIO
from datetime import datetime
from abc import ABC, abstractmethod
from llm_platform.helpers.model_config import ModelConfig
from llm_platform.services.files import BaseFile, DocumentFile, TextDocumentFile, PDFDocumentFile, ExcelDocumentFile, MediaFile, ImageFile, AudioFile, VideoFile
import json
import base64


class FunctionCall:
    def __init__(self, 
                 id: str,
                 name: str,
                 arguments: Dict | List[Dict],
                 call_id: str=None
                ):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.call_id = call_id

    @classmethod
    def from_openai(cls, tool_call):
        return cls(id=tool_call.id, 
                   name=tool_call.name, 
                   arguments=str(tool_call.arguments),
                   call_id=tool_call.call_id
                   )
    
    def to_openai(self) -> Dict:
        return {"id": self.id, 
                "call_id": self.call_id,
                "name": self.name, 
                "arguments": self.arguments,
                "type": "function_call",
            }
    
    def to_anthropic(self) -> Dict:
        return {"id": self.id,
                "name": self.name, 
                "input": json.loads(self.arguments),
                "type": "tool_use",
                }

    def __str__(self):
        return f"Id: {self.id}; Function: {self.name}, Arguments: {self.arguments}"

class FunctionResponse:
    def __init__(self, 
                 name: str, 
                 response: Dict,
                 id: str=None,
                 call_id: str=None
                 ):
        self.name = name
        self.id = id
        self.call_id = call_id
        if isinstance(response, dict):
            self.response = response
        else:
            self.response = {"text": response}
        self.files = []

        self._parse_response()

    def _parse_response(self):
        if "files" in self.response:
            assert isinstance(self.response["files"], list), "`files` must be a list"
            """
            Example `files` key in response_dict:
                "files": [{
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "format": "png",
                                            "data": image as base64 string,
                                        },
                                    }],
            """
            for file in self.response["files"]:
                assert "type" in file, "File must have a 'type' key"
                if file["type"] == "image":
                    assert "source" in file, "File must have a 'source' key"
                    assert "type" in file["source"], "File source must have a 'type' key"
                    assert "format" in file["source"], "File source must have a 'format' key"
                    if file["source"]["type"] == "base64":
                        assert "data" in file["source"], "File source must have a 'type' data"
                        self.files.append(ImageFile.from_base64(base64_str=file["source"]["data"],
                                                                file_name=f"image.{file['source']['format']}")
                                        )
            self.response.pop("files")

    def to_openai(self) -> Dict:
        if self.files:
            print("WARNING: Files are not supported in function responses for OpenAI")
        return {
                "type":"function_call_output", 
                "call_id": self.call_id, 
                #"name": self.name, 
                "output": json.dumps(self.response)
        }
    
    def to_anthropic(self) -> Dict:
        output = {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": [{"type": "text", 
                         "text": json.dumps(self.response)}] 
        }

        if self.files:
            for file in self.files:
                if isinstance(file, ImageFile):
                    new_content_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{file.extension}",
                            "data": file.base64,
                        }
                    }
                    output["content"].append(new_content_block)
        return output

    def __str__(self):
        return f"Id: {self.id}; Function: {self.name}, Response: {json.dumps(self.response)}"

class ThinkingResponse:
    def __init__(self, 
                 content: str,
                 id: str=None,
                 ):
        self.content = content
        self.id = id

    def to_anthropic(self) -> Dict:
        return {
                "type": "thinking",
                "thinking": self.content,
                "signature": self.id 
            }

    def __str__(self):
        return self.content

class Message:
    def __init__(self, 
                 role: str, 
                 content: str, 
                 thinking_responses: List[ThinkingResponse]=[],
                 usage: Dict=None, 
                 files: List[BaseFile] = [],
                 function_calls: List[FunctionCall]=[],
                 function_responses: List[FunctionResponse]=[],
                 ):
        
        assert role in ["user", "assistant", "function"] #the only possible roles
        self.role = role
        self.content = content
        self.thinking_responses = thinking_responses
        self.timestamp = datetime.now()
        self.files = files
        self.usage = usage
        self.function_calls = function_calls
        self.function_responses = function_responses

    def __str__(self):
        return f"{self.role}: {self.content};" + \
                "\n".join([str(thinking_response) for thinking_response in self.thinking_responses]) + "\n" + \
                "\n".join([str(function_call) for function_call in self.function_calls]) + "\n\n" + \
                "\n".join([str(function_response) for function_response in self.function_responses])

class Conversation:
    def __init__(self, 
                 messages: List[Message] = [], 
                 system_prompt: str = None):
        """
        Initializes a Conversation instance.

        Parameters:
        ----------
        messages : List[Message], optional
            A list of Message objects representing the conversation history. If not provided, 
            an empty list is created by default.
        
        system_prompt : str, optional
            A string representing the system prompt for the conversation. If not provided, it defaults to None.
        
        Attributes:
        ----------
        messages : List[Message]
            Stores the messages of the conversation.
        
        model_config : ModelConfig
            An instance of `ModelConfig` that manages model-specific configurations.
        
        system_prompt : str or None
            The initial system prompt used in the conversation.
        """

        self.messages = messages if messages is not None else []
        #self.model_config = ModelConfig()
        self.system_prompt = system_prompt

    def __str__(self):
        return "\n".join([str(message) for message in self.messages])

    def clear(self):
        self.messages.clear()

    @property
    def usage_total(self) -> Dict:
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "costs": 0
        }
        for message in self.messages:
            if message.usage:
                total_usage["prompt_tokens"] += message.usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += message.usage.get("completion_tokens", 0)
                total_usage["costs"] += message.usage.get("costs", 0)
        return total_usage

    @property
    def usage_last(self) -> Dict:
        if self.messages and self.messages[-1].usage:
            return self.messages[-1].usage
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "costs": 0
        }
        
    def save_to_json(self) -> Dict:
        """
        Serialize the conversation to a JSON-compatible dictionary.
        
        Returns:
        -------
        Dict
            A dictionary representation of the conversation that can be serialized to JSON.
        """
        # Convert conversation to a serializable dictionary
        data = {
            'system_prompt': self.system_prompt,
            'messages': []
        }
        
        for message in self.messages:
            message_data = {
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'usage': message.usage,
                'thinking_responses': [
                    {'content': tr.content, 'id': tr.id}
                    for tr in message.thinking_responses
                ],
                'function_calls': [
                    {
                        'id': fc.id,
                        'name': fc.name,
                        'arguments': fc.arguments
                    }
                    for fc in message.function_calls
                ],
                'function_responses': [
                    {
                        'name': fr.name,
                        'id': fr.id,
                        'response': fr.response,
                        'files': self._serialize_files(fr.files)
                    }
                    for fr in message.function_responses
                ],
                'files': self._serialize_files(message.files)
            }
            data['messages'].append(message_data)
        
        return data
        
    def _serialize_files(self, files: List[BaseFile]) -> List[Dict]:
        """
        Helper method to serialize different types of files.
        
        Parameters:
        ----------
        files : List[BaseFile]
            List of file objects to serialize
            
        Returns:
        -------
        List[Dict]
            List of serialized file dictionaries
        """
        serialized_files = []
        
        for file in files:
            file_data = {
                'name': file.name,
                'type': type(file).__name__
            }

            # Handle different file types
            if isinstance(file, TextDocumentFile):
                file_data['text'] = file.text
                
            elif isinstance(file, PDFDocumentFile):
                file_data['base64'] = file.base64
                file_data['text'] = file.text
                file_data['number_of_pages'] = file.number_of_pages
                
            elif isinstance(file, ExcelDocumentFile):
                file_data['base64'] = file.base64
                file_data['text'] = file.text
                
            elif isinstance(file, ImageFile):
                file_data['base64'] = file.base64
                file_data['extension'] = file.extension
                
            elif isinstance(file, AudioFile):
                file_data['base64'] = file.base64
                file_data['extension'] = file.extension
                
            elif isinstance(file, VideoFile):
                if hasattr(file, 'base64') and file.base64:
                    file_data['base64'] = file.base64
                file_data['extension'] = file.extension
                
            elif isinstance(file, MediaFile):
                file_data['base64'] = file.base64
                file_data['extension'] = file.extension
            
            serialized_files.append(file_data)
            
        return serialized_files

    @classmethod
    def read_from_json(cls, data:Dict) -> 'Conversation':
        """
        Read a conversation from a JSON-compatible dictionary.
        
        Parameters:
        ----------
        data : Dict
            The dictionary containing the serialized conversation.
            
        Returns:
        -------
        Conversation
            A new Conversation object loaded from the dictionary.
        """
        messages = []
        for msg_data in data.get('messages', []):
            # Restore files
            files = cls._deserialize_files(msg_data.get('files', []))
            
            # Restore thinking responses
            thinking_responses = [
                ThinkingResponse(content=tr_data['content'], id=tr_data.get('id'))
                for tr_data in msg_data.get('thinking_responses', [])
            ]
            
            # Restore function calls
            function_calls = [
                FunctionCall(id=fc_data['id'], name=fc_data['name'], arguments=fc_data['arguments'])
                for fc_data in msg_data.get('function_calls', [])
            ]
            
            # Restore function responses
            function_responses = []
            for fr_data in msg_data.get('function_responses', []):
                fr = FunctionResponse(name=fr_data['name'], response=fr_data['response'], id=fr_data.get('id'))
                
                # Restore files in function responses
                fr.files = cls._deserialize_files(fr_data.get('files', []))
                function_responses.append(fr)
            
            # Create message
            message = Message(
                role=msg_data['role'],
                content=msg_data['content'],
                thinking_responses=thinking_responses,
                usage=msg_data.get('usage'),
                files=files,
                function_calls=function_calls,
                function_responses=function_responses
            )
            
            # Restore timestamp
            if 'timestamp' in msg_data:
                message.timestamp = datetime.fromisoformat(msg_data['timestamp'])
            
            messages.append(message)
        
        return cls(messages=messages, system_prompt=data.get('system_prompt'))
        
    @classmethod
    def _deserialize_files(cls, file_data_list: List[Dict]) -> List[BaseFile]:
        """
        Helper method to deserialize different types of files.
        
        Parameters:
        ----------
        file_data_list : List[Dict]
            List of serialized file dictionaries
            
        Returns:
        -------
        List[BaseFile]
            List of reconstructed file objects
        """
        files = []
        
        for file_data in file_data_list:
            if 'type' not in file_data or 'name' not in file_data:
                continue
                
            file_type = file_data['type']
            file_name = file_data['name']
            
            if file_type == 'TextDocumentFile' and 'text' in file_data:
                files.append(TextDocumentFile(text=file_data['text'], name=file_name))
                
            elif file_type == 'ImageFile' and file_data.get('base64'):
                files.append(ImageFile.from_base64(file_data['base64'], file_name))
                
            elif file_type == 'PDFDocumentFile' and file_data.get('base64'):
                pdf_bytes = base64.b64decode(file_data['base64'])
                files.append(PDFDocumentFile.from_bytes(pdf_bytes, file_name))
                
            elif file_type == 'AudioFile' and file_data.get('base64'):
                audio_bytes = base64.b64decode(file_data['base64'])
                files.append(AudioFile.from_bytes(audio_bytes, file_name))
                
            elif file_type == 'ExcelDocumentFile' and file_data.get('base64'):
                excel_bytes = base64.b64decode(file_data['base64'])
                files.append(ExcelDocumentFile.from_bytes(excel_bytes, file_name))
                
            elif file_type == 'VideoFile' and file_data.get('base64'):
                video_bytes = base64.b64decode(file_data['base64'])
                files.append(MediaFile.from_bytes(video_bytes, file_name))
                
            elif file_type == 'MediaFile' and file_data.get('base64'):
                media_bytes = base64.b64decode(file_data['base64'])
                files.append(MediaFile.from_bytes(media_bytes, file_name))
                
        return files