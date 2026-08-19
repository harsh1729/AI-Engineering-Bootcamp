"""Add pgvector extension and per-provider embedding tables for RAG retrieval."""

from typing import Sequence, Union

from alembic import op

revision: str = "009_pgvector_embeddings"
down_revision: Union[str, None] = "008_document_content_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_embedding_table(table_name: str, dimensions: int) -> None:
    op.execute(
        f"""
        CREATE TABLE {table_name} (
            chunk_id TEXT PRIMARY KEY,
            document_id UUID NOT NULL,
            chunk_index INTEGER NOT NULL,
            source_filename TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE INDEX ix_{table_name}_document_id
        ON {table_name} (document_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX ix_{table_name}_embedding_hnsw
        ON {table_name}
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    _create_embedding_table("doc_pg_embeddings_openai", 1536)
    _create_embedding_table("doc_pg_embeddings_voyage", 1024)
    _create_embedding_table("doc_pg_embeddings_cohere", 1536)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS doc_pg_embeddings_cohere")
    op.execute("DROP TABLE IF EXISTS doc_pg_embeddings_voyage")
    op.execute("DROP TABLE IF EXISTS doc_pg_embeddings_openai")
