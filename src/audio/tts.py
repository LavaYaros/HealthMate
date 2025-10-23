"""
Text-to-Speech utility module for HealthMate.
Provides simple interface for converting medical guidance to speech.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional
from src.audio.tts_adapter import get_tts_adapter
from src.logging.logger import setup_logger

logger = setup_logger(__name__)

# Singleton TTS instance
_tts_instance = None


def get_tts():
    """Get or create singleton TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = get_tts_adapter()
    return _tts_instance


def text_to_speech(
    text: str,
    output_path: Optional[str] = None,
    use_medical_formatting: bool = True
) -> bytes:
    """
    Convert text to speech audio.
    
    Args:
        text: Text to convert to speech
        output_path: Optional path to save audio file
        use_medical_formatting: If True, applies medical-optimized SSML formatting
        
    Returns:
        Audio data as bytes
        
    Example:
        >>> audio = text_to_speech("Apply pressure to stop bleeding.")
        >>> # Save to file
        >>> audio = text_to_speech("Check for breathing.", output_path="instruction.mp3")
    """
    tts = get_tts()
    
    # Use SSML for medical instructions if Polly is the provider
    if use_medical_formatting and hasattr(tts, 'create_medical_ssml'):
        ssml = tts.create_medical_ssml(text)
        return tts.synthesize_ssml(ssml, output_path)
    else:
        return tts.synthesize(text, output_path)


def ssml_to_speech(ssml: str, output_path: Optional[str] = None) -> bytes:
    """
    Convert SSML markup to speech audio.
    
    Args:
        ssml: SSML markup to convert
        output_path: Optional path to save audio file
        
    Returns:
        Audio data as bytes
        
    Example:
        >>> ssml = '<speak><prosody rate="slow">Speak slowly.</prosody></speak>'
        >>> audio = ssml_to_speech(ssml)
    """
    tts = get_tts()
    return tts.synthesize_ssml(ssml, output_path)

