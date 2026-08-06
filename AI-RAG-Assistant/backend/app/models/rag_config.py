from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    SENTENCE = "sentence"
    HEADER_AWARE = "header_aware"


class EmbeddingProviderType(StrEnum):
    OPENAI = "openai"
    VOYAGE = "voyage"
    COHERE = "cohere"


class VectorStoreType(StrEnum):
    CHROMA = "chroma"


class RagOptions(BaseModel):
    """Indexing and retrieval settings for the RAG pipeline."""

    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    embedding_provider: EmbeddingProviderType = EmbeddingProviderType.OPENAI
    vector_store: VectorStoreType = VectorStoreType.CHROMA


DEFAULT_RAG_OPTIONS = RagOptions()


class RagOptionsCatalogItem(BaseModel):
    value: str
    label: str
    enabled: bool = True


class RagOptionsCatalogResponse(BaseModel):
    chunking_strategies: list[RagOptionsCatalogItem]
    embedding_providers: list[RagOptionsCatalogItem]
    vector_stores: list[RagOptionsCatalogItem]
    embedding_models: dict[str, str]
    defaults: RagOptions
