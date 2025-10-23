"""
Amazon Polly Text-to-Speech implementation.
Supports both standard and neural voices with SSML markup.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import Optional, List, Dict
from contextlib import closing

from src.audio.tts_adapter import TTSAdapter
from src.config.settings import get_settings
from src.config.aws_config import configure_aws
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class PollyTTS(TTSAdapter):
    """
    Amazon Polly Text-to-Speech implementation.
    
    Features:
    - Neural and standard voice engines
    - SSML support for advanced speech control
    - Multiple voice options (Joanna, Matthew, etc.)
    - Adjustable speech rate via SSML
    """
    
    def __init__(self):
        """Initialize Amazon Polly client."""
        # Configure AWS credentials
        configure_aws()
        
        # Initialize Polly client
        self.client = boto3.client('polly', region_name=settings.AWS_REGION)
        self.voice_id = settings.POLLY_VOICE_ID
        self.engine = settings.POLLY_ENGINE
        
        logger.info(f"Polly TTS initialized: voice={self.voice_id}, engine={self.engine}")
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> bytes:
        """
        Convert text to speech using Amazon Polly.
        
        Args:
            text: Plain text to convert to speech
            output_path: Optional path to save audio file (MP3 format)
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        try:
            # Apply speech rate if configured
            if settings.VOICE_SPEED != 1.0:
                # Convert to SSML with prosody rate
                rate_percent = int(settings.VOICE_SPEED * 100)
                ssml_text = f'<speak><prosody rate="{rate_percent}%">{text}</prosody></speak>'
                return self.synthesize_ssml(ssml_text, output_path)
            
            # Request speech synthesis
            response = self.client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=self.voice_id,
                Engine=self.engine
            )
            
            # Get audio stream
            if "AudioStream" in response:
                with closing(response["AudioStream"]) as stream:
                    audio_data = stream.read()
                    
                    # Save to file if path provided
                    if output_path:
                        with open(output_path, 'wb') as file:
                            file.write(audio_data)
                        logger.info(f"Audio saved to: {output_path}")
                    
                    logger.info(f"Synthesized {len(text)} characters to {len(audio_data)} bytes")
                    return audio_data
            else:
                raise Exception("No audio stream in Polly response")
                
        except (BotoCoreError, ClientError) as error:
            logger.error(f"Polly synthesis error: {error}")
            raise
    
    def synthesize_ssml(self, ssml: str, output_path: Optional[str] = None) -> bytes:
        """
        Convert SSML markup to speech using Amazon Polly.
        
        Args:
            ssml: SSML markup text
            output_path: Optional path to save audio file (MP3 format)
            
        Returns:
            Audio data as bytes (MP3 format)
            
        Example SSML:
            <speak>
                <prosody rate="slow">This is spoken slowly.</prosody>
                <break time="1s"/>
                <emphasis level="strong">This is emphasized.</emphasis>
            </speak>
        """
        try:
            # Request speech synthesis with SSML
            response = self.client.synthesize_speech(
                Text=ssml,
                TextType='ssml',
                OutputFormat='mp3',
                VoiceId=self.voice_id,
                Engine=self.engine
            )
            
            # Get audio stream
            if "AudioStream" in response:
                with closing(response["AudioStream"]) as stream:
                    audio_data = stream.read()
                    
                    # Save to file if path provided
                    if output_path:
                        with open(output_path, 'wb') as file:
                            file.write(audio_data)
                        logger.info(f"Audio saved to: {output_path}")
                    
                    logger.info(f"Synthesized SSML to {len(audio_data)} bytes")
                    return audio_data
            else:
                raise Exception("No audio stream in Polly response")
                
        except (BotoCoreError, ClientError) as error:
            logger.error(f"Polly SSML synthesis error: {error}")
            raise
    
    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available Polly voices.
        
        Returns:
            List of voice dictionaries with Id, Name, Gender, LanguageCode
        """
        try:
            response = self.client.describe_voices(
                Engine=self.engine,
                LanguageCode='en-US'  # Filter for English voices
            )
            
            voices = []
            for voice in response.get('Voices', []):
                voices.append({
                    'Id': voice['Id'],
                    'Name': voice['Name'],
                    'Gender': voice['Gender'],
                    'LanguageCode': voice['LanguageCode']
                })
            
            logger.info(f"Found {len(voices)} available voices")
            return voices
            
        except (BotoCoreError, ClientError) as error:
            logger.error(f"Error fetching voices: {error}")
            return []
    
    def create_medical_ssml(self, text: str) -> str:
        """
        Create SSML optimized for medical first-aid instructions.
        Adds appropriate pauses and emphasis for clarity.
        
        Args:
            text: Plain text medical instruction
            
        Returns:
            SSML formatted text
        """
        # Apply speech rate
        rate_percent = int(settings.VOICE_SPEED * 100)
        
        # Add pauses after sentences for clarity
        text_with_pauses = text.replace('. ', '.<break time="0.5s"/> ')
        
        # Wrap in SSML with prosody
        ssml = f'''<speak>
            <prosody rate="{rate_percent}%">
                {text_with_pauses}
            </prosody>
        </speak>'''
        
        return ssml


def get_polly_tts() -> PollyTTS:
    """Get PollyTTS instance."""
    return PollyTTS()
