from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


load_dotenv()



class Settings(BaseSettings):
    """Settings for the application."""

    # LLM Configuration
    LLM_MODEL: str = "gpt-4o-mini"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", None)
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.95

    # RAG Configuration
    TOP_K: int = 3
    DEBUG_MODE: bool = True # Set to True to print debug information to console

    # Memory configuration
    MEMORY_TOKEN_LIMIT: int = 512 # Amount of tokens after which conversation is summarized
    MAX_MESSAGES: int = 4 # Amount of messages after which conversation is summarized
    KEEP_RECENT_MESSAGES: int = 2 # Amount of recent messages to keep after summarization

    class Config:
        env_file = ".env"

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
