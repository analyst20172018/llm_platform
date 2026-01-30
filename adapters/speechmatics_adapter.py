from .adapter_base import AdapterBase
import os
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from typing import List, Tuple, Dict, BinaryIO, Callable, Union, Tuple, Optional
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import AudioFile
from loguru import logger
from llm_platform.types import AdditionalParameters

class SpeechmaticsAdapter(AdapterBase):
    
    def __init__(self):
        super().__init__()  
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")

    def convert_conversation_history_to_adapter_format(self, the_conversation: Conversation):
        raise NotImplementedError("Not implemented yet")

    def voice_to_text(self, audio_file: Tuple[str, BinaryIO], language: str="en"):
        """
            audio_file should be a tuple (file name, bytes)
        """

        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=self.api_key,
        )

        # Define transcription parameters
        transcription_config = { 
                                    "language": language,
                                    # Find out more about entity detection here:
                                    # https://docs.speechmatics.com/features/entities#enable-entity-metadata
                                    "enable_entities": True,
                                    "operating_point": "enhanced",
                                    "diarization": "speaker"
                                }

        conf = {
            "type": "transcription",
            "transcription_config": transcription_config,
        }

        # Open the client using a context manager
        with BatchClient(settings) as client:
            try:
                job_id = client.submit_job(
                    #audio=PATH_TO_FILE,
                    audio=audio_file,
                    transcription_config=conf,
                )
                #print(f"job {job_id} submitted successfully, waiting for transcript")

                # Note that in production, you should set up notifications instead of polling. 
                # Notifications are described here: https://docs.speechmatics.com/features-other/notifications
                transcript = client.wait_for_completion(job_id, transcription_format="txt")
                # To see the full output, try setting transcription_format="json-v2".
                #print(transcript)
                return transcript
            except Exception as e:
                print(f"Error: {str(e)}")
                return str(e)
        
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
        
        # Check that the only file is audio file.
        audio_file = files[0]
        if not isinstance(audio_file, AudioFile):
            logger.error("The provided file is not audio. I need only audio file to transcribe it.")
            message = Message(role="assistant", content="The provided file is not audio. I need only audio file to transcribe it.", usage={})
            the_conversation.messages.append(message)
            return message

        language = additional_parameters.get("language", "en")
        if len(language) != 2:
            logger.error("Language code must be 2 characters long. For examle 'en' for English or 'cs' for Czech")
            message = Message(role="assistant", content="Language code must be 2 characters long. For examle 'en' for English or 'cs' for Czech.", usage={})
            the_conversation.messages.append(message)
            return message

        transcript = self.voice_to_text(
            (audio_file.name, audio_file.bytes_io), 
            language
        )

        message = Message(role="assistant", content=transcript, usage={})
        the_conversation.messages.append(message)
        
        return message

    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        NotImplementedError("Not implemented yet")

    def request_llm_with_functions(self, 
                                   model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[Callable],
                                   tool_output_callback: Callable=None, 
                                   additional_parameters: AdditionalParameters | None = None,
                                   **kwargs):
        raise NotImplementedError("Not implemented yet")

    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")
