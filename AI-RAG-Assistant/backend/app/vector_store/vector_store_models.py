from pydantic import BaseModel

from app.embeddings.embedding_models import EmbeddingVector


class VectorStoreRecord(BaseModel):
    """One chunk and its embedding ready for vector storage."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    embedding: EmbeddingVector


class VectorQueryResult(BaseModel):
    """One chunk returned from a similarity search."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    distance: float
