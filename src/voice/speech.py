import io
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load the project .env even when this module is imported directly from a Streamlit page.
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# IMPORTANT: Groq's Turbo model supports multilingual transcription but NOT audio translation.
# whisper-large-v3 supports both transcription and translation to English.
VOICE_MODEL = "whisper-large-v3"

def translate_audio_to_english(audio_bytes: bytes, filename: str = "voice.wav") -> str:
    """Convert recorded speech directly into English text using Groq Whisper.

    The Groq translations endpoint always returns English. We intentionally use
    whisper-large-v3 here because whisper-large-v3-turbo does not support the
    translations task.
    """
    if not audio_bytes:
        raise RuntimeError("The microphone recording is empty.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add GROQ_API_KEY=your_key to the project .env file and restart Streamlit."
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename or "voice.wav"

    try:
        result = client.audio.translations.create(
            model=VOICE_MODEL,
            file=audio_file,
            response_format="json",
            temperature=0,
        )
    except Exception as exc:
        message = str(exc)
        raise RuntimeError(f"Groq speech-to-English request failed: {message}") from exc

    text = (result.text or "").strip()
    if not text:
        raise RuntimeError("Groq returned no speech text. Please record your voice again, speaking clearly for a few seconds.")
    return text
