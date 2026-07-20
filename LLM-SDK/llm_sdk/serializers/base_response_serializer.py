from abc import ABC, abstractmethod
from typing import Any
from llm_sdk.models import LLMResponse


class BaseResponseSerializer(ABC):

    @abstractmethod
    def serialize(self, response:Any) -> LLMResponse:
        """
        Convert specific LLM provider responce into the format
        of internal LLMReponse.
        """
        pass