from abc import ABC, abstractmethod
from typing import Any

from llm_sdk.models import LLMResponseChunk


class BaseResponseChunkSerializer(ABC):

    @abstractmethod
    def serialize(
        self,
        event: Any,
    ) -> LLMResponseChunk | None:
        """
        Convert a provider-specific streaming event
        into an LLMResponseChunk.
        """
        raise NotImplementedError