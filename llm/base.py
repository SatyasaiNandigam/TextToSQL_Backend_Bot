from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def get_llm(self):
        pass
    
    