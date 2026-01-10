import io
import os
from typing import Any, BinaryIO, Dict, Tuple

from elevenlabs.client import ElevenLabs

MODEL_NAME = "scribe_v2"


class ElenenlabsAdapter:
    def __init__(self):
        self.api_key = os.getenv("ELEVEN_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)


    def voice_to_text(
        self,
        audio_file: BinaryIO,
        audio_format: str,
        language: str = "eng",
        diarized: bool = True,
    ) -> Dict[str, Any]:
        audio_file = self._ensure_seekable(audio_file)

        transcription = self.client.speech_to_text.convert(
            file=audio_file,
            model_id=MODEL_NAME,
            tag_audio_events=True,
            language_code=language,
            diarize=diarized,
        )

        return {
            "text": transcription.text,
            "words": transcription.words if diarized else None,
            "language": language,
            "model": MODEL_NAME,
        }


    @staticmethod
    def format_diarization(response: Dict[str, Any]) -> str:
        """
        Formats the diarized response dictionary into a string with marked speakers.

        Args:
            response (Dict[str, Any]): The dictionary returned by voice_to_text.

        Returns:
            str: The formatted text with marked speakers, or the plain text if no words found.
        """
        words = response.get("words")
        if not words:
            return response.get("text", "")

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

