from app.config import resolve_embedding_model
from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.cohere_embedding_provider import CohereEmbeddingProvider
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from app.embeddings.voyage_embedding_provider import VoyageEmbeddingProvider
from app.models.rag_config import EmbeddingProviderType


class EmbeddingProviderFactory:
    """Creates embedding provider implementations."""

    _PROVIDERS: dict[EmbeddingProviderType, type[BaseEmbeddingProvider]] = {
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProviderType.VOYAGE: VoyageEmbeddingProvider,
        EmbeddingProviderType.COHERE: CohereEmbeddingProvider,
    }

    @classmethod
    def create(cls, provider: EmbeddingProviderType) -> BaseEmbeddingProvider:
        provider_class = cls._PROVIDERS.get(provider)
        if provider_class is None:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        return provider_class(model=resolve_embedding_model(provider.value))


# Backward-compatible alias used by earlier factory wiring.
EmbeddingFactory = EmbeddingProviderFactory
