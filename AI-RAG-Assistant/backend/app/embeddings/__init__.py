"""Embedding domain models for the RAG pipeline.

Provider implementations and API orchestration live in later phases.
Callers should depend on these models, not vendor-specific types.
"""

from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import (
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "OpenAIEmbeddingProvider",
]
