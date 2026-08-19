import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import QDRANT_API_KEY, QDRANT_URL
from app.services.document_exceptions import VectorDimensionMismatchError
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord


def _score_to_distance(score: float) -> float:
    """Convert Qdrant cosine similarity to ascending distance (lower is better)."""
    return 1.0 - score


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUIDs; app chunk IDs are uuid4().hex."""
    return str(uuid.UUID(hex=chunk_id))


def _point_id_to_chunk_id(point_id: Any) -> str:
    return uuid.UUID(str(point_id)).hex


def _record_payload(record: VectorStoreRecord) -> dict[str, Any]:
    return {
        "document_id": record.document_id,
        "chunk_index": record.chunk_index,
        "source_filename": record.source_filename,
        "text": record.text,
    }


class QdrantVectorStore(BaseVectorStore):
    """Vector store backed by a Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name is required for Qdrant vector store.")

        if client is None:
            if not QDRANT_URL:
                raise ValueError("QDRANT_URL is required for Qdrant vector store.")
            client_kwargs: dict[str, Any] = {"url": QDRANT_URL}
            if QDRANT_API_KEY:
                client_kwargs["api_key"] = QDRANT_API_KEY
            client = QdrantClient(**client_kwargs)

        self._client = client
        self._collection_name = collection_name
        self._expected_dimension = expected_dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._expected_dimension is None:
            return
        if self._client.collection_exists(self._collection_name):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._expected_dimension,
                distance=Distance.COSINE,
            ),
        )

    def _validate_dimension(self, vector: list[float], *, context: str) -> None:
        if self._expected_dimension is None:
            return
        actual = len(vector)
        if actual != self._expected_dimension:
            raise VectorDimensionMismatchError(
                f"Embedding dimension {actual} does not match Qdrant collection "
                f"'{self._collection_name}' expected dimension {self._expected_dimension} "
                f"({context})."
            )

    def upsert(self, records: list[VectorStoreRecord]) -> None:
        if not records:
            return

        points: list[PointStruct] = []
        for record in records:
            self._validate_dimension(
                record.embedding.embedding,
                context=f"chunk_id={record.chunk_id}",
            )
            points.append(
                PointStruct(
                    id=_chunk_id_to_point_id(record.chunk_id),
                    vector=record.embedding.embedding,
                    payload=_record_payload(record),
                )
            )

        self._client.upsert(collection_name=self._collection_name, points=points)

    def query(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorQueryResult]:
        self._validate_dimension(query_embedding, context="query embedding")

        query_filter = None
        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return self._map_query_response(response)

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def _map_query_response(self, response: Any) -> list[VectorQueryResult]:
        points = getattr(response, "points", None) or []
        if not points:
            return []

        results: list[VectorQueryResult] = []
        for point in points:
            payload = point.payload or {}
            score = point.score
            if score is None:
                continue

            results.append(
                VectorQueryResult(
                    chunk_id=_point_id_to_chunk_id(point.id),
                    document_id=payload["document_id"],
                    chunk_index=payload["chunk_index"],
                    text=payload.get("text", ""),
                    source_filename=payload["source_filename"],
                    distance=_score_to_distance(float(score)),
                )
            )

        results.sort(key=lambda result: result.distance)
        return results
