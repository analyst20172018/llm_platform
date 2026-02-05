from .adapter_base import AdapterBase
import os
from typing import List, Tuple, Dict, BinaryIO, Callable, Union, Tuple, Optional
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import AudioFile
from loguru import logger
from llm_platform.types import AdditionalParameters
import assemblyai as aai
from assemblyai import api, types
import time

class AssemblyAIAdapter:
    
    def __init__(self):
        super().__init__()  
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    def voice_to_text(self, audio_file: AudioFile, additional_parameters: Dict) -> str:

        language_detection = additional_parameters.get("language_detection", False)
        language = additional_parameters.get("language", "en")
        diarized = additional_parameters.get("diarized", True)
        speakers_expected = additional_parameters.get("speakers_expected", None)

        config_parameters = {
            "speech_models": ["universal-2"],
            "speaker_labels": diarized,
        }
        if language_detection:
            config_parameters["language_detection"] = True
        else:
            config_parameters["language_code"] = language

        if speakers_expected is not None:
            config_parameters["speakers_expected"] = speakers_expected

        config = aai.TranscriptionConfig(**config_parameters)
        aai.settings.http_timeout = 120.0  # seconds - timeout for uploading large files

        transcriber = aai.Transcriber(config=config) 
        transcript = transcriber.submit(audio_file.file_bytes)
        transcript_id = transcript.id

        client = aai.Client.get_default()
        while True:
            resp = api.get_transcript(client.http_client, transcript_id)
            if resp.status in (types.TranscriptStatus.completed, types.TranscriptStatus.error):
                break
            time.sleep(aai.settings.polling_interval)

        transcript = aai.Transcript.from_response(client=client, response=resp)

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
