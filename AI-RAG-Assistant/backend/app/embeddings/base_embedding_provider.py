from abc import ABC, abstractmethod

from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingResponse


class BaseEmbeddingProvider(ABC):
    """Strategy interface for converting text into vector embeddings.

    Implementations wrap a specific model or API (OpenAI, local models, etc.)
    and return provider-agnostic response objects. They must not perform
    chunking, document parsing, vector storage, or retrieval.
    """

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResponse:
        """Return the embedding vector for a single text input."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        """Return embedding vectors for multiple text inputs.

        Results must preserve input order. Batch-level usage is returned once
        on the response, not duplicated per vector.
        """
