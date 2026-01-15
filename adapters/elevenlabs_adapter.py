import io
import os
from typing import Any, BinaryIO, Dict, Tuple, List, Callable
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import (AudioFile, BaseFile, DocumentFile,
                                         TextDocumentFile, PDFDocumentFile,
                                         ExcelDocumentFile, WordDocumentFile, PowerPointDocumentFile, 
                                         MediaFile, ImageFile, VideoFile)
from elevenlabs.client import ElevenLabs
from loguru import logger
from llm_platform.types import AdditionalParameters

class ElevenLabsAdapter:
    def __init__(self):
        self.api_key = os.getenv("ELEVEN_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)


    def voice_to_text(
        self,
        model_name: str,
        audio_file: BinaryIO,
        language: str = "eng",
        diarized: bool = True,
        tag_audio_events: bool = True,
        num_speakers: int = None,
    ) -> Dict[str, Any]:
        audio_file = self._ensure_seekable(audio_file)

        transcription = self.client.speech_to_text.convert(
            file=audio_file,
            model_id=model_name,
            tag_audio_events=tag_audio_events,
            language_code=language,
            diarize=diarized,
            num_speakers=num_speakers,
        )

        """
        The type of the output (transcription):
            * **language_code** *(string)* — Detected language code (e.g., `"eng"` for English)
            * **language_probability** *(double)* — Confidence score of language detection *(0–1)*
            * **text** *(string)* — Raw transcription text
            * **words** *(list[object])* — Words with timing information if diarized is True 
            * **channel_index** *(int | null)* — Channel index for this transcript (multichannel audio)
            * **additional_formats** *(list[object?] | null)* — Requested additional transcript formats
            * **transcription_id** *(string | null)* — Transcription ID in the response
            * **entities** *(list[object] | null)* — Detected entities (text, type, character positions)
        """
        return transcription

    @staticmethod
    def format_diarization(transcription) -> str:
        """
        Formats the diarized response dictionary into a string with marked speakers.

        Args:
            response (Dict[str, Any]): The dictionary returned by voice_to_text.

        Returns:
            str: The formatted text with marked speakers, or the plain text if no words found.
        """
        words = getattr(transcription, "words", None)
        if not words:
            return getattr(transcription, "text", "")

        diarized_output = []
        current_speaker = None
        current_text_parts = []

        for word in words:
            speaker_id = getattr(word, "speaker_id", None)
            text = getattr(word, "text", "")

            # If speaker changes, flush the previous segment
            if speaker_id != current_speaker:
                if current_speaker is not None:
                    # Join and strip to clean up spacing
                    segment = "".join(current_text_parts).strip()
                    if segment:
                        diarized_output.append(f"[{current_speaker}]: {segment}")

                current_speaker = speaker_id
                current_text_parts = []

            current_text_parts.append(text)

        # Flush the final segment
        if current_speaker is not None:
            segment = "".join(current_text_parts).strip()
            if segment:
                diarized_output.append(f"[{current_speaker}]: {segment}")

        return "\n\n".join(diarized_output)


    def _ensure_seekable(self, audio_file: BinaryIO) -> BinaryIO:
        if audio_file.seekable():
            return audio_file
        data = audio_file.read()
        if not data:
            raise ValueError("Audio stream is empty.")
        return io.BytesIO(data)

    def request_llm(self, 
                    model: str, 
                    the_conversation: Conversation, 
                    temperature: int=0,  
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
        diarized = additional_parameters.get("diarized", True)
        tag_audio_events = additional_parameters.get("tag_audio_events", True)
        num_speakers = additional_parameters.get("num_speakers", None)

        transcription = self.voice_to_text(
            model,
            audio_file.bytes_io, 
            language, 
            diarized, 
            tag_audio_events, 
            num_speakers
        )

        response = self.format_diarization(transcription)
        usage = {}
        
        message = Message(role="assistant", content=response, usage=usage)
        the_conversation.messages.append(message)
        
        return message
