from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI
import asyncio
import time
from typing import List, Dict, Union

from .settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

class LLMObject:
    """
    Unified LLM interface supporting multiple providers (OpenAI, Bedrock).
    Uses LangChain's chat model interface for consistent message handling.
    """
    def __init__(self, MODEL_ID=None, provider=None, temperature=None, top_p=None):
        self.provider = provider or settings.PROVIDER
        self.model_id = MODEL_ID or (settings.LLM_MODEL if self.provider == 'openai' else settings.MODEL_ID)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.top_p = top_p if top_p is not None else settings.LLM_TOP_P
        self.bedrock_region = settings.BEDROCK_REGION
        self.bedrock_access_key = settings.BEDROCK_ACCESS_KEY
        self.bedrock_secret_key = settings.BEDROCK_SECRET_KEY
        
        if not self.model_id:
            raise ValueError("MODEL_ID must be set in the environment or passed to LLMObject.")
        
        if self.provider == 'bedrock':
            self.llm = self._init_bedrock_llm()
        elif self.provider == 'openai':
            self.llm = self._init_openai_llm()
        else:
            raise ValueError(f"Unknown PROVIDER: {self.provider}. Use 'bedrock' or 'openai'.")

    def _init_openai_llm(self):
        """Initialize OpenAI LLM using LangChain's ChatOpenAI."""
        return ChatOpenAI(
            model=self.model_id,
            api_key=settings.openai_api_key,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def _init_bedrock_llm(self):
        """Initialize AWS Bedrock LLM using LangChain's ChatBedrockConverse."""
        if not (self.bedrock_region and self.bedrock_access_key and self.bedrock_secret_key):
            raise ValueError("BEDROCK_REGION, BEDROCK_ACCESS_KEY, and BEDROCK_SECRET_KEY must be set for Bedrock provider.")
        
        return ChatBedrockConverse(
            model=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            region_name=self.bedrock_region,
            aws_access_key_id=self.bedrock_access_key,
            aws_secret_access_key=self.bedrock_secret_key,
        )

    def invoke(self, messages: List[Union[Dict[str, str], object]]):
        """
        Synchronous invocation with LangChain message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        
        Returns:
            Response object with .content attribute
        """
        max_retries = 5
        delay = 1
        
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                if "ThrottlingException" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise RuntimeError("API throttled: max retries exceeded.")
                else:
                    raise

    async def ainvoke(self, messages: List[Union[Dict[str, str], object]]):
        """
        Async invocation for non-blocking operation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        
        Returns:
            Response object with .content attribute
        """
        return await asyncio.to_thread(self.invoke, messages)


