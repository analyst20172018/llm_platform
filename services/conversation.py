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
                ):
        self.id = id
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_openai(cls, tool_call):
        return cls(id=tool_call.id, 
                   name=tool_call.function.name, 
                   arguments=str(tool_call.function.arguments),
                   )
    
    def to_openai(self) -> Dict:
        return {"id": self.id, 
            "function": 
                {"name": self.name, 
                 "arguments": self.arguments},
            "type": "function",
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
                 id: str=None
                 ):
        self.name = name
        self.id = id
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
                "role":"tool", 
                "tool_call_id": self.id, 
                "name": self.name, 
                "content": json.dumps(self.response)
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
        Save the conversation to a file.
        
        Parameters:
        ----------
        filepath : str
            The path to the file where the conversation should be saved.
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
                        'files': [
                            {
                                'name': file.name,
                                'type': type(file).__name__,
                                'base64': file.base64 if hasattr(file, 'base64') else None
                            }
                            for file in fr.files
                        ]
                    }
                    for fr in message.function_responses
                ],
                'files': [
                    {
                        'name': file.name,
                        'type': type(file).__name__,
                        'base64': file.base64 if hasattr(file, 'base64') else None,
                        'text': file.text if hasattr(file, 'text') else None
                    }
                    for file in message.files
                ]
            }
            data['messages'].append(message_data)
        
        #with open(filepath, 'w', encoding='utf-8') as f:
        #    json.dump(data, f, indent=2)
        return data

    @classmethod
    def read_from_json(cls, data:Dict) -> 'Conversation':
        """
        Read a conversation from a file.
        
        Parameters:
        ----------
        filepath : str
            The path to the file containing the saved conversation.
            
        Returns:
        -------
        Conversation
            A new Conversation object loaded from the file.
        """
        #with open(filepath, 'r', encoding='utf-8') as f:
        #    data = json.load(f)
        
        messages = []
        for msg_data in data.get('messages', []):
            # Restore files
            files = []
            for file_data in msg_data.get('files', []):
                file_type = file_data['type']
                if file_type == 'TextDocumentFile' and 'text' in file_data:
                    files.append(TextDocumentFile(text=file_data['text'], name=file_data['name']))
                elif file_type == 'ImageFile' and file_data.get('base64'):
                    files.append(ImageFile.from_base64(file_data['base64'], file_data['name']))
                elif file_type == 'PDFDocumentFile' and file_data.get('base64'):
                    pdf_bytes = base64.b64decode(file_data['base64'])
                    files.append(PDFDocumentFile.from_bytes(pdf_bytes, file_data['name']))
                elif file_type == 'AudioFile' and file_data.get('base64'):
                    audio_bytes = base64.b64decode(file_data['base64'])
                    files.append(AudioFile.from_bytes(audio_bytes, file_data['name']))
                elif file_type == 'ExcelDocumentFile' and file_data.get('base64'):
                    excel_bytes = base64.b64decode(file_data['base64'])
                    files.append(ExcelDocumentFile.from_bytes(excel_bytes, file_data['name']))
            
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
                fr.files = []
                for file_data in fr_data.get('files', []):
                    file_type = file_data['type']
                    if file_type == 'ImageFile' and file_data.get('base64'):
                        fr.files.append(ImageFile.from_base64(file_data['base64'], file_data['name']))
                
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
