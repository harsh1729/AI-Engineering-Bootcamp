import pytest
from unittest.mock import MagicMock

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
from app.vector_store.pinecone_vector_store import PineconeVectorStore


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

    def test_create_pinecone_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_pinecone_cls = MagicMock(return_value=mock_client)
        monkeypatch.setattr(
            "app.vector_store.pinecone_vector_store.Pinecone",
            mock_pinecone_cls,
        )
        monkeypatch.setattr(
            "app.vector_store.pinecone_vector_store.PINECONE_API_KEY",
            "test-pinecone-key",
        )
        monkeypatch.setattr(
            "app.factories.vector_store_factory.resolve_pinecone_index",
            lambda provider: f"index-{provider}",
        )
        monkeypatch.setattr(
            "app.factories.vector_store_factory.resolve_pinecone_dimension",
            lambda provider: 1536 if provider == "openai" else 1024,
        )

        store = VectorStoreFactory.create(
            VectorStoreType.PINECONE,
            embedding_provider=EmbeddingProviderType.OPENAI,
        )

        assert isinstance(store, PineconeVectorStore)
        mock_client.Index.assert_called_once_with("index-openai")

    def test_create_pinecone_store_requires_embedding_provider(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider is required"):
            VectorStoreFactory.create(VectorStoreType.PINECONE)

    def test_create_pinecone_store_uses_provider_specific_index(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_client = MagicMock()
        monkeypatch.setattr(
            "app.vector_store.pinecone_vector_store.Pinecone",
            MagicMock(return_value=mock_client),
        )
        monkeypatch.setattr(
            "app.vector_store.pinecone_vector_store.PINECONE_API_KEY",
            "test-pinecone-key",
        )
        monkeypatch.setattr(
            "app.factories.vector_store_factory.resolve_pinecone_index",
            lambda provider: {
                "openai": "ai-rag-openai-1536",
                "voyage": "ai-rag-voyage-1024",
                "cohere": "ai-rag-cohere-1536",
            }[provider],
        )
        monkeypatch.setattr(
            "app.factories.vector_store_factory.resolve_pinecone_dimension",
            lambda provider: 1536 if provider == "openai" else 1024,
        )

        openai_store = VectorStoreFactory.create(
            VectorStoreType.PINECONE,
            embedding_provider=EmbeddingProviderType.OPENAI,
        )
        voyage_store = VectorStoreFactory.create(
            VectorStoreType.PINECONE,
            embedding_provider=EmbeddingProviderType.VOYAGE,
        )

        assert openai_store._index_name == "ai-rag-openai-1536"
        assert voyage_store._index_name == "ai-rag-voyage-1024"
        assert openai_store is not voyage_store
