from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os

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