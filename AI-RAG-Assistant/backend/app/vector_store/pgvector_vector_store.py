from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import (
    PGVECTOR_TABLE_BY_PROVIDER,
    sync_database_url,
)
from app.services.document_exceptions import VectorDimensionMismatchError
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

_ALLOWED_TABLES = frozenset(PGVECTOR_TABLE_BY_PROVIDER.values())


def _validate_table_name(table_name: str) -> str:
    if table_name not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid pgvector table name: {table_name}")
    return table_name


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


@lru_cache
def _get_sync_engine() -> Engine:
    return create_engine(sync_database_url(), pool_pre_ping=True)


class PgVectorVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL pgvector tables."""

    def __init__(
        self,
        table_name: str,
        expected_dimension: int | None = None,
        engine: Engine | None = None,
    ) -> None:
        if not table_name:
            raise ValueError("table_name is required for pgvector vector store.")

        self._table_name = _validate_table_name(table_name)
        self._expected_dimension = expected_dimension
        self._engine = engine or _get_sync_engine()

    def _validate_dimension(self, vector: list[float], *, context: str) -> None:
        if self._expected_dimension is None:
            return
        actual = len(vector)
        if actual != self._expected_dimension:
            raise VectorDimensionMismatchError(
                f"Embedding dimension {actual} does not match pgvector table "
                f"'{self._table_name}' expected dimension {self._expected_dimension} "
                f"({context})."
            )

    def upsert(self, records: list[VectorStoreRecord]) -> None:
        if not records:
            return

        upsert_sql = text(
            f"""
            INSERT INTO {self._table_name} (
                chunk_id, document_id, chunk_index, source_filename, text, embedding
            )
            VALUES (
                :chunk_id,
                CAST(:document_id AS uuid),
                :chunk_index,
                :source_filename,
                :text,
                CAST(:embedding AS vector)
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                chunk_index = EXCLUDED.chunk_index,
                source_filename = EXCLUDED.source_filename,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """
        )

        with self._engine.begin() as conn:
            for record in records:
                self._validate_dimension(
                    record.embedding.embedding,
                    context=f"chunk_id={record.chunk_id}",
                )
                conn.execute(
                    upsert_sql,
                    {
                        "chunk_id": record.chunk_id,
                        "document_id": record.document_id,
                        "chunk_index": record.chunk_index,
                        "source_filename": record.source_filename,
                        "text": record.text,
                        "embedding": _vector_literal(record.embedding.embedding),
                    },
                )

    def query(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorQueryResult]:
        self._validate_dimension(query_embedding, context="query embedding")

        where_clause = ""
        params: dict[str, object] = {
            "query_embedding": _vector_literal(query_embedding),
            "limit": limit,
        }
        if document_ids:
            where_clause = "WHERE document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = document_ids

        query_sql = text(
            f"""
            SELECT
                chunk_id,
                document_id::text AS document_id,
                chunk_index,
                source_filename,
                text,
                embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM {self._table_name}
            {where_clause}
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
            """
        )

        with self._engine.begin() as conn:
            rows = conn.execute(query_sql, params).mappings().all()

        return [
            VectorQueryResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                source_filename=row["source_filename"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def delete_document(self, document_id: str) -> None:
        delete_sql = text(
            f"""
            DELETE FROM {self._table_name}
            WHERE document_id = CAST(:document_id AS uuid)
            """
        )

        with self._engine.begin() as conn:
            conn.execute(delete_sql, {"document_id": document_id})
