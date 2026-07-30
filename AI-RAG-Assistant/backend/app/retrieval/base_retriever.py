from abc import ABC, abstractmethod

from app.retrieval.retrieval_models import RetrievalRequest, RetrievalResponse


class BaseRetriever(ABC):
    """Strategy interface for retrieving relevant document chunks for a query."""

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return chunks relevant to `request`."""
