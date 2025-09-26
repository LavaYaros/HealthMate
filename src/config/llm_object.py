from langchain_aws import ChatBedrockConverse
import openai
import asyncio
import time

from .settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

class LLMObject:
    def __init__(self, MODEL_ID=None, provider=None):
        self.provider = provider or settings.PROVIDER
        self.model_id = MODEL_ID or settings.MODEL_ID
        self.bedrock_region = settings.BEDROCK_REGION
        self.bedrock_access_key = settings.BEDROCK_ACCESS_KEY
        self.bedrock_secret_key = settings.BEDROCK_SECRET_KEY
        if not self.model_id:
            raise ValueError("MODEL_ID must be set in the environment or passed to LLMObject.")
        if self.provider == 'bedrock':
            self._init_bedrock_llm()
        elif self.provider == 'openai':
            self._init_openai_llm()
        else:
            raise ValueError(f"Unknown PROVIDER: {self.provider}. Use 'local', 'bedrock', or 'openai'.")

    def _init_openai_llm(self):
        # OpenAI LLM initialization for openai>=1.0.0
        self.openai_api_key = get_settings().OPENAI_API_KEY
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        return "OpenAI LLM loaded"

    def _init_bedrock_llm(self):
        # Placeholder for AWS Bedrock Qwen LLM initialization
        if not (self.bedrock_region and self.bedrock_access_key and self.bedrock_secret_key):
            raise ValueError("BEDROCK_REGION, BEDROCK_ACCESS_KEY, and BEDROCK_SECRET_KEY must be set for Bedrock provider.")
        return f"AWS Bedrock Qwen LLM provider for {self.model_id} in {self.bedrock_region}"

    def _generate_sync(self, prompt: str, temperature: float=0.1, max_tokens: int=settings.MAX_GENERATION_TOKENS) -> str:
        """Synchronous generation implementation (kept for run_in_executor)."""        
        if self.provider == "bedrock":
            llm = ChatBedrockConverse(
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                provider=self.provider,
                region_name=self.bedrock_region,
                aws_access_key_id=self.bedrock_access_key,
                aws_secret_access_key=self.bedrock_secret_key,
            )
            max_retries = 5
            delay = 1
            for attempt in range(max_retries):
                try:
                    response = llm.invoke([{"role": "user", "content": prompt}])
                    if hasattr(response, "content"):
                        return response.content
                    if isinstance(response, dict) and "content" in response:
                        return response["content"]
                    return str(response)
                except Exception as e:
                    if "ThrottlingException" in str(e):
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
            logger.warning("Bedrock response missing 'content' field.")
            raise RuntimeError("Bedrock API throttled: max retries exceeded.")  
        
        elif self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            return response.choices[0].message.content
        return f"[No generate method for {self.model_id} via {self.provider}]"

    async def generate(self, prompt: str) -> str:
        """Async wrapper around sync generation. Uses thread executor for blocking providers."""
        # For providers that offer async APIs you can implement direct awaits here.
        return await asyncio.to_thread(self._generate_sync, prompt)


