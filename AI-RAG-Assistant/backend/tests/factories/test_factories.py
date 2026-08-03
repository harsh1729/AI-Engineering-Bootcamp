import pytest

from app.chunking.character_chunker import CharacterChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.config import CHUNK_SIZE
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from app.factories.chunking_factory import ChunkingFactory
from app.factories.embedding_factory import EmbeddingFactory
from app.factories.vector_store_factory import VectorStoreFactory
from app.models.rag_config import ChunkingStrategy, EmbeddingProviderType, VectorStoreType
from app.vector_store.chroma_vector_store import ChromaVectorStore


class TestChunkingFactory:
    def test_create_recursive_chunker(self) -> None:
        chunker = ChunkingFactory.create(ChunkingStrategy.RECURSIVE)

        assert isinstance(chunker, RecursiveChunker)
        assert chunker._chunk_size == CHUNK_SIZE

    def test_create_character_chunker(self) -> None:
        chunker = ChunkingFactory.create(ChunkingStrategy.CHARACTER)

        assert isinstance(chunker, CharacterChunker)
        assert chunker._chunk_size == CHUNK_SIZE


class TestEmbeddingFactory:
    def test_create_openai_provider(self) -> None:
        provider = EmbeddingFactory.create(EmbeddingProviderType.OPENAI)

        assert isinstance(provider, OpenAIEmbeddingProvider)


class TestVectorStoreFactory:
    def test_create_chroma_store(self) -> None:
        store = VectorStoreFactory.create(VectorStoreType.CHROMA)

        assert isinstance(store, ChromaVectorStore)
