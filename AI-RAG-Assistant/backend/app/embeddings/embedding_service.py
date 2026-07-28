from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingResponse


class EmbeddingService:
    """Orchestrates text embedding via a configured provider.

    Caching, retries, metrics, and batching policy are handled elsewhere or
    added in later phases. This service only delegates to the injected
    BaseEmbeddingProvider implementation.
    """

    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self._provider = provider

    def embed(self, text: str) -> EmbeddingResponse:
        """Return the embedding for a single text input."""
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        """Return embeddings for multiple text inputs."""
        return self._provider.embed_batch(texts)
