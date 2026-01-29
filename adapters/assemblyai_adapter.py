from .adapter_base import AdapterBase
import os
from typing import List, Tuple, Dict, BinaryIO, Callable, Union, Tuple, Optional
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import AudioFile
from loguru import logger
from llm_platform.types import AdditionalParameters
import assemblyai as aai

class AssemblyAIAdapter:
    
    def __init__(self):
        super().__init__()  
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    def voice_to_text(self, audio_file: AudioFile, additional_parameters: Dict) -> str:

        language_detection = additional_parameters.get("language_detection", False)
        language = additional_parameters.get("language", "en")
        diarized = additional_parameters.get("diarized", True)
        num_speakers = additional_parameters.get("num_speakers", None)

        config_parameters = {
            "speech_models": ["universal"],
            "speaker_labels": diarized,
        }
        if language_detection:
            config_parameters["language_detection"] = True
        else:
            config_parameters["language_code"] = language

        if num_speakers is not None:
            config_parameters["speaker_count"] = num_speakers

        config = aai.TranscriptionConfig(**config_parameters)

        transcript = aai.Transcriber(config=config).transcribe(audio_file.file_bytes)

        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            return f"Transcription failed: {transcript.error}"
        
        # Convert transcript to string
        output_transcript = ""
        if diarized and transcript.utterances is not None:
            for utterance in transcript.utterances:
                output_transcript += f"Speaker {utterance.speaker}: {utterance.text}\n"
        else:
            output_transcript = transcript.text
        
        return output_transcript
        
    def request_llm(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    tool_output_callback: Callable=None,
                    additional_parameters: AdditionalParameters | None = None,
                    **kwargs) -> Message:
        if additional_parameters is None:
            additional_parameters = {}

        if kwargs:
            logger.warning("Passing request parameters via **kwargs is deprecated; use additional_parameters.")
            for key, value in kwargs.items():
                additional_parameters.setdefault(key, value)
        
        # Get files from the last message. I expect only one file - audio file for transcription
        files = getattr(the_conversation.messages[-1], "files", [])
        if len(files) != 1:
            logger.error(f"There are {len(files)} files in the last message. I expect only one audio file to transcribe it.")
            message = Message(role="assistant", content=f"There are {len(files)} files in the last message. I expect only one audio file to transcribe it.", usage={})
            the_conversation.messages.append(message)
            return message
        
        transcript = self.voice_to_text(files[0], additional_parameters)

        message = Message(role="assistant", content=transcript, usage={})
        the_conversation.messages.append(message)
        
        return message
