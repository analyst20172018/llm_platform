from .adapter_base import AdapterBase
import os
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from typing import List, Tuple, Dict, BinaryIO, Callable, Union, Tuple, Optional
from llm_platform.services.conversation import Conversation, Message

class SpeechmaticsAdapter(AdapterBase):
    
    def __init__(self):
        super().__init__()  
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")

    def convert_conversation_history_to_adapter_format(self, the_conversation: Conversation):
        raise NotImplementedError("Not implemented yet")

    def request_llm(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    temperature: int=0, 
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={},
                    **kwargs) -> Message:
        raise NotImplementedError("Not implemented yet")

    def voice_to_text(self, audio_file: Tuple[str, BinaryIO], language: str="en", transcription_config: Dict=None):
        """
            audio_file should be a tuple (file name, bytes)
        """

        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=self.api_key,
        )

        # Define transcription parameters
        if transcription_config is None:
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
        
    def generate_image(self, prompt: str, size: str, quality:str, n=1):
        NotImplementedError("Not implemented yet")

    def request_llm_with_functions(self, 
                                   model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[Callable],
                                   tool_output_callback: Callable=None, 
                                   additional_parameters: Dict={},
                                   **kwargs):
        raise NotImplementedError("Not implemented yet")

    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")
