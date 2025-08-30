"""
LLM Adapter for HealthMate: Configurable interface to the selected LLM provider/model.
Uses LangChain for orchestration. Prompts are scoped to home first aid, with citation and safety rules.
"""

import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import get_settings

# Prompt templates
SYSTEM_MAIN_PROMPT = (
    "You are HealthMate, an AI assistant for home-based first aid. "
    "Limit scope to home first aid only. "
    "Always provide step-by-step guidance and cite sources from the knowledge base. "
    "Include contraindication checks and escalate red flags (e.g., when to call emergency services). "
    "Do not diagnose or prescribe. If context is insufficient, ask a brief clarifying question before advising. "
    "Explicitly advise to seek emergency help for severe bleeding, chest pain, stroke signs, anaphylaxis, etc."
)

MESSAGE_TEMPLATE = "Question: {query}\nContext: {context}"

class LLMObject:
    def __init__(self):
        settings = get_settings()
        self.model_name = settings.LLM_MODEL
        self.api_key = settings.API_KEY
        self.temperature = settings.LLM_TEMPERATURE
        self.top_p = settings.LLM_TOP_P
        # Only OpenAI provider for PoC; can be extended via config
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def generate(self, query: str, context: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_MAIN_PROMPT),
            HumanMessage(content=MESSAGE_TEMPLATE.format(query=query, context=context)),
        ]
        response = self.llm(messages)
        return response.content

# Minimal test script
if __name__ == "__main__":
    llm = LLMObject()
    test_query = "How do I treat a minor burn at home?"
    test_context = "Chunked KB content about burns, with citations."
    answer = llm.generate(test_query, test_context)
    print("Answer:\n", answer)
