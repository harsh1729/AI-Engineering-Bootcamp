from abc import ABC, abstractmethod
from typing import Generator
from models import LLMRequest,LLMResponse,LLMResponseChunk

class LLMProvider(ABC):

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:
        """Generate a response stream for the given prompt."""
        pass