import pytest

from app.chunking.character_chunker import CharacterChunker
from app.chunking.header_aware_chunker import HeaderAwareChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.config import CHUNK_SIZE
from app.embeddings.cohere_embedding_provider import CohereEmbeddingProvider
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from app.embeddings.voyage_embedding_provider import VoyageEmbeddingProvider
from app.factories.chunking_factory import ChunkingFactory
from app.factories.embedding_factory import EmbeddingFactory, EmbeddingProviderFactory
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

    def test_create_sentence_chunker(self) -> None:
        chunker = ChunkingFactory.create(ChunkingStrategy.SENTENCE)

        assert isinstance(chunker, SentenceChunker)
        assert chunker._chunk_size == CHUNK_SIZE

    def test_create_header_aware_chunker(self) -> None:
        chunker = ChunkingFactory.create(ChunkingStrategy.HEADER_AWARE)

        assert isinstance(chunker, HeaderAwareChunker)
        assert chunker._chunk_size == CHUNK_SIZE


class TestEmbeddingFactory:
    def test_create_openai_provider(self) -> None:
        provider = EmbeddingFactory.create(EmbeddingProviderType.OPENAI)

        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_create_voyage_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "app.embeddings.voyage_embedding_provider.VOYAGE_API_KEY",
            "test-voyage-key",
        )
        provider = EmbeddingProviderFactory.create(EmbeddingProviderType.VOYAGE)

        assert isinstance(provider, VoyageEmbeddingProvider)

    def test_create_cohere_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "app.embeddings.cohere_embedding_provider.COHERE_API_KEY",
            "test-cohere-key",
        )
        provider = EmbeddingProviderFactory.create(EmbeddingProviderType.COHERE)

        assert isinstance(provider, CohereEmbeddingProvider)


class TestVectorStoreFactory:
    def test_create_chroma_store(self) -> None:
        store = VectorStoreFactory.create(VectorStoreType.CHROMA)

        assert isinstance(store, ChromaVectorStore)
