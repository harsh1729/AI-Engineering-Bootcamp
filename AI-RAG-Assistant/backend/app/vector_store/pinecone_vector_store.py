from typing import Any

from pinecone import Pinecone

from app.config import PINECONE_API_KEY, PINECONE_NAMESPACE
from app.services.document_exceptions import PineconeDimensionMismatchError
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord


def _score_to_distance(score: float) -> float:
    """Convert Pinecone cosine similarity to ascending distance (lower is better)."""
    return 1.0 - score


def _record_metadata(record: VectorStoreRecord) -> dict[str, Any]:
    return {
        "document_id": record.document_id,
        "chunk_index": record.chunk_index,
        "source_filename": record.source_filename,
        "text": record.text,
    }


class PineconeVectorStore(BaseVectorStore):
    """Vector store backed by a Pinecone index."""

    def __init__(
        self,
        client: Pinecone | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        if not index_name:
            raise ValueError("index_name is required for Pinecone vector store.")

        if client is None:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is required for Pinecone vector store.")
            client = Pinecone(api_key=PINECONE_API_KEY)

        self._index_name = index_name
        self._index = client.Index(index_name)
        self._namespace = PINECONE_NAMESPACE if namespace is None else namespace
        self._expected_dimension = expected_dimension

    def _validate_dimension(self, vector: list[float], *, context: str) -> None:
        if self._expected_dimension is None:
            return
        actual = len(vector)
        if actual != self._expected_dimension:
            raise PineconeDimensionMismatchError(
                f"Embedding dimension {actual} does not match Pinecone index "
                f"'{self._index_name}' expected dimension {self._expected_dimension} "
                f"({context})."
            )

    def upsert(self, records: list[VectorStoreRecord]) -> None:
        if not records:
            return

        for record in records:
            self._validate_dimension(
                record.embedding.embedding,
                context=f"chunk_id={record.chunk_id}",
            )

        self._index.upsert(
            vectors=[
                {
                    "id": record.chunk_id,
                    "values": record.embedding.embedding,
                    "metadata": _record_metadata(record),
                }
                for record in records
            ],
            namespace=self._namespace,
        )

    def query(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorQueryResult]:
        self._validate_dimension(query_embedding, context="query embedding")

        query_kwargs: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": limit,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if document_ids:
            query_kwargs["filter"] = {"document_id": {"$in": document_ids}}

        response = self._index.query(**query_kwargs)
        return self._map_query_response(response)

    def delete_document(self, document_id: str) -> None:
        self._index.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=self._namespace,
        )

    def _map_query_response(self, response: Any) -> list[VectorQueryResult]:
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        if not matches:
            return []

        results: list[VectorQueryResult] = []
        for match in matches:
            metadata = getattr(match, "metadata", None) or match["metadata"]
            score = getattr(match, "score", None)
            if score is None:
                score = match["score"]
            chunk_id = getattr(match, "id", None) or match["id"]

            results.append(
                VectorQueryResult(
                    chunk_id=chunk_id,
                    document_id=metadata["document_id"],
                    chunk_index=metadata["chunk_index"],
                    text=metadata.get("text", ""),
                    source_filename=metadata["source_filename"],
                    distance=_score_to_distance(float(score)),
                )
            )

        results.sort(key=lambda result: result.distance)
        return results
