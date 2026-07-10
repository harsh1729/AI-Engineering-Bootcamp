from abc import ABC, abstractmethod
from typing import Any

from models import LLMResponseChunk


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