from abc import ABC, abstractmethod
from typing import Any

from models import LLMMessage


class LLMMessageSerializer(ABC):

    @abstractmethod
    def serialize_messages(
        self,
        messages: list[LLMMessage],
    ) -> Any:
        """
        Convert internal LLM messages into the format
        expected by a specific LLM provider.
        """
        pass