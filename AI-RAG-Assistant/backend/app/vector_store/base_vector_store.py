from abc import ABC, abstractmethod

from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord


class BaseVectorStore(ABC):
    """Strategy interface for persisting and searching embedded document chunks."""

    @abstractmethod
    def upsert(self, records: list[VectorStoreRecord]) -> None:
        """Insert or update the supplied chunk records."""

    @abstractmethod
    def query(self, query_embedding: list[float], limit: int) -> list[VectorQueryResult]:
        """Return the closest stored chunks to `query_embedding`.

        Results are ordered by ascending distance (lower is better).
        """

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all stored chunks belonging to `document_id`."""
