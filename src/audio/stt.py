"""Speech-to-Text module using OpenAI Whisper API."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI
from src.config.settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

class WhisperSTT:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe audio file to text."""
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        logger.info(f"Transcribed audio: {len(transcript.text)} chars")
        return transcript.text

def get_stt() -> WhisperSTT:
    """Get STT instance."""
    return WhisperSTT()