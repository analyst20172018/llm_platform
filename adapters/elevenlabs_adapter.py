from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os
from typing import List, Tuple, Dict, BinaryIO, Callable

load_dotenv()

def get_voices():
    """
        Get the list of all the voices
        The list of all the voices with the description can be found here: https://elevenlabs.io/app/voice-lab
    """
    client = ElevenLabs(
        api_key=os.getenv("ELEVEN_API_KEY")
    )
    response = client.voices.get_all()
    return response

def generate(text: str, voice):
    """
        Generate audio from text
        
        Args:
            text (str): The text to generate audio from
            voice: The voice to use for the audio generation. Voice can be the str (name of the voice) or the object:
                response = get_voices()
                voice = response.voices[0]
    """
    client = ElevenLabs(
        api_key=os.getenv("ELEVEN_API_KEY")
    )

    audio = client.generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    return audio

def voice_to_text(audio_file: BinaryIO, language: str="eng"):
    client = ElevenLabs(
        api_key=os.getenv("ELEVEN_API_KEY"),
    )

    try:

        transcription = client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
            tag_audio_events=True, # Tag audio events like laughter, applause, etc.
            language_code=language, # Language of the audio file, according to ISO 639. If set to None, the model will detect the language automatically.
            diarize=True, # Whether to annotate who is speaking
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return transcription.text