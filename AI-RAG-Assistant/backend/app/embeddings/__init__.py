"""Embedding domain models for the RAG pipeline.

Provider implementations and API orchestration live in later phases.
Callers should depend on these models, not vendor-specific types.
"""

from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.cohere_embedding_provider import CohereEmbeddingProvider
from app.embeddings.embedding_models import (
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from app.embeddings.voyage_embedding_provider import VoyageEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "CohereEmbeddingProvider",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
]
