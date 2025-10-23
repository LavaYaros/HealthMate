from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from typing import Optional


load_dotenv()


class Settings(BaseSettings):
    """Settings for the application."""

    # LLM Configuration

    PROVIDER: str = os.getenv("PROVIDER", "openai")  # 'bedrock', or 'openai'

    LLM_MODEL: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.95

    MODEL_ID: Optional[str] = os.getenv("MODEL_ID")
    BEDROCK_REGION: Optional[str] = os.getenv("BEDROCK_REGION")
    BEDROCK_ACCESS_KEY: Optional[str] = os.getenv("BEDROCK_ACCESS_KEY")
    BEDROCK_SECRET_KEY: Optional[str] = os.getenv("BEDROCK_SECRET_KEY")

    EMBEDDING_MODEL: Optional[str] = "text-embedding-3-small"  # OpenAI embedding model

    WHISPER_MODEL: str = "whisper-1"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # TTS Configuration
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "polly")  # polly | elevenlabs | azure
    POLLY_VOICE_ID: str = os.getenv("POLLY_VOICE_ID", "Joanna")  # Joanna, Matthew, etc.
    POLLY_ENGINE: str = os.getenv("POLLY_ENGINE", "neural")  # standard | neural
    VOICE_SPEED: float = float(os.getenv("VOICE_SPEED", "1.0"))  # 0.25 - 2.0
    
    # RAG Configuration
    TOP_K: int = 5  # Number of passages to retrieve
    SIMILARITY_THRESHOLD: float = 0.5  # Minimum similarity score (0-1) for relevance filtering
    DEDUP_THRESHOLD: float = 0.85  # Text similarity threshold for deduplication (0-1)
    MAX_CONTEXT_TOKENS: int = 2000  # Maximum tokens for context passed to LLM
    DEBUG_MODE: bool = True # Set to True to print debug information to console

    # Memory configuration
    MAX_MESSAGES_FOR_SUMMARIZATION: int = 6 # Amount of messages after which conversation is summarized
    MAX_MESSAGES_FOR_MEMORY: int = 20 # Amount of messages to keep in memory
    MEMORY_PATH: str = "src/memory/storage/" # Path to store conversation memory files
    
    # Hugging Face token (for Gradio deployment)
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from .env

@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached instance of Settings.

    Uses LRU (Least Recently Used) caching to:
    - Avoid creating new Settings objects on every call
    - Ensure all parts of the app use the same Settings instance

    The cache is cleared when the application is restarted.
    """
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print("DEBUG_MODE:", settings.DEBUG_MODE)
