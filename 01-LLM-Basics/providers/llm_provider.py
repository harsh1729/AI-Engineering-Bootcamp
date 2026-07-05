from abc import ABC, abstractmethod
from models import LLMRequest

class LLMProvider(ABC):

    @abstractmethod
    def generate(self, request: LLMRequest):
        """Generate a response for the given prompt."""
        pass