from abc import ABC, abstractmethod
from typing import Any

from llm_sdk.models import LLMRequest


class BaseRequestSerializer(ABC):

    @abstractmethod
    def serialize(
        self,
        request: LLMRequest,
    ) -> Any:
        """
        Convert an internal LLMRequest into the format
        expected by a specific LLM provider.
        """
        raise NotImplementedError