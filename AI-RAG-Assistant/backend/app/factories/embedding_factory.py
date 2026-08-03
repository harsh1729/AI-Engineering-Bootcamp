from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from app.models.rag_config import EmbeddingProviderType


class EmbeddingFactory:
    """Creates embedding provider implementations."""

    _PROVIDERS: dict[EmbeddingProviderType, type[BaseEmbeddingProvider]] = {
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
    }

    @classmethod
    def create(cls, provider: EmbeddingProviderType) -> BaseEmbeddingProvider:
        provider_class = cls._PROVIDERS.get(provider)
        if provider_class is None:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        return provider_class()
