"""
Text-to-Speech adapter interface for HealthMate.
Provides unified interface for multiple TTS providers (Amazon Polly, ElevenLabs, Azure).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abc import ABC, abstractmethod
from typing import Optional, BinaryIO

from src.config.settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class TTSAdapter(ABC):
    """Base class for Text-to-Speech providers."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as bytes
        """
        pass
    
    @abstractmethod
    def synthesize_ssml(self, ssml: str, output_path: Optional[str] = None) -> bytes:
        """
        Convert SSML (Speech Synthesis Markup Language) to speech audio.
        
        Args:
            ssml: SSML markup to convert
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as bytes
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> list:
        """
        Get list of available voices for the provider.
        
        Returns:
            List of voice identifiers/names
        """
        pass


def get_tts_adapter(provider: Optional[str] = None) -> TTSAdapter:
    """
    Factory function to get TTS adapter based on configuration.
    
    Args:
        provider: Optional provider override (polly | elevenlabs | azure)
        
    Returns:
        TTSAdapter instance
    """
    provider = provider or settings.TTS_PROVIDER
    
    if provider == "polly":
        from src.audio.providers.polly_tts import PollyTTS
        logger.info("Initializing Amazon Polly TTS")
        return PollyTTS()
    elif provider == "elevenlabs":
        # Future implementation
        raise NotImplementedError("ElevenLabs TTS not yet implemented")
    elif provider == "azure":
        # Future implementation
        raise NotImplementedError("Azure Speech TTS not yet implemented")
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


if __name__ == "__main__":
    # Test TTS adapter initialization
    print(f"TTS Provider: {settings.TTS_PROVIDER}")
    tts = get_tts_adapter()
    print(f"✓ TTS adapter initialized: {type(tts).__name__}")
