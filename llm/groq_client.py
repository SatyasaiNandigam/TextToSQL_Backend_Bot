from langchain_groq import ChatGroq
from llm.base import BaseLLMClient
from core.settings import settings


class GroqClient(BaseLLMClient):

    def __init__(self, model=settings.GROQ_MODEL, temperature=0):
        self.model = model
        self.temperature = temperature

    def get_llm(self):
        return ChatGroq(
            model=self.model,
            temperature=self.temperature,
            groq_api_key=settings.GROQ_API_KEY,
        )