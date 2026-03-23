from langchain_openai import ChatOpenAI
from llm.base import BaseLLMClient
from core.settings import settings


class OpenAIClient(BaseLLMClient):

    def __init__(self, model=settings.DEFAULT_LLM_MODEL, temperature=0):
        self.model = model
        self.temperature = temperature

    def get_llm(self):
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=settings.OPENAI_API_KEY,
        )