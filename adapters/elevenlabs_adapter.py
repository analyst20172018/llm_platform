from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os
from typing import List, Tuple, Dict, BinaryIO, Callable
import logging

load_dotenv()

class ElenenlabsAdapter:
    def __init__(self, logging_level=logging.INFO):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVEN_API_KEY")
        )
    
    def get_voices(self):
        """
            Get the list of all the voices
            The list of all the voices with the description can be found here: https://elevenlabs.io/app/voice-lab
        """
        response = self.client.voices.get_all()
        return response
    
    def generate(self, text: str, voice):
        """
            Generate audio from text
            
            Args:
                text (str): The text to generate audio from
                voice: The voice to use for the audio generation. Voice can be the str (name of the voice) or the object:
                    response = get_voices()
                    voice = response.voices[0]
        """
        audio = self.client.generate(
            text=text,
            voice=voice,
            model="eleven_multilingual_v3"
        )
        return audio
    
    def _transcribe(self, audio_file: BinaryIO, language: str="eng"):
        try:
            transcription = self.client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1_experimental", # The ID of the model to use for transcription, currently only ‘scribe_v1’ and ‘scribe_v1_experimental’ are available.
                tag_audio_events=True, # Tag audio events like laughter, applause, etc.
                language_code=language, # Language of the audio file, according to ISO 639. If set to None, the model will detect the language automatically.
                diarize=True, # Whether to annotate who is speaking
            )
        except Exception as e:
            print(f"Error: {str(e)}")
        
        return transcription

    def voice_to_text(self, audio_file: BinaryIO, language: str="eng", diarized: bool=True):
        transcription = self._transcribe(
                audio_file=audio_file,
                language=language, # Language of the audio file, according to ISO 639. If set to None, the model will detect the language automatically.
            )
        
        if not diarized:
            return transcription.text
    
        # Get words and speakers
        output_text = ""
        current_speaker = None
        current_text = ""
        for word in transcription.words:
            if current_speaker != word.speaker_id:
                if current_text != "":
                    if output_text != "":
                        output_text += "\n"
                    output_text += f"{current_speaker}: {current_text}"
                
                current_speaker = word.speaker_id
                current_text = ""

            current_text += word.text

        if current_text != "":
            if output_text != "":
                output_text += "\n"
            output_text += f"{current_speaker}: {current_text}"
        
        return output_text