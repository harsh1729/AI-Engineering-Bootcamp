import pytest

from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)
from app.services.rag_pipeline_builder import (
    _get_shared_chunking_service,
    _get_shared_vector_store,
    build_document_ingestion_service,
)


class TestSharedChunkingService:
    def test_returns_same_instance_for_same_strategy(self) -> None:
        _get_shared_chunking_service.cache_clear()

        recursive_a = _get_shared_chunking_service(ChunkingStrategy.RECURSIVE)
        recursive_b = _get_shared_chunking_service(ChunkingStrategy.RECURSIVE)
        character = _get_shared_chunking_service(ChunkingStrategy.CHARACTER)

        assert recursive_a is recursive_b
        assert recursive_a is not character

    def test_build_document_ingestion_service_uses_shared_chunking_service(self) -> None:
        _get_shared_chunking_service.cache_clear()

        service = build_document_ingestion_service(
            RagOptions(chunking_strategy=ChunkingStrategy.CHARACTER)
        )
        cached = _get_shared_chunking_service(ChunkingStrategy.CHARACTER)

        assert service._chunking_service is cached


class TestSharedVectorStore:
    def test_returns_same_instance_for_same_store_and_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _get_shared_vector_store.cache_clear()
        created: list[object] = []

        def fake_create(
            store_type: VectorStoreType,
            *,
            embedding_provider: EmbeddingProviderType,
        ) -> object:
            instance = object()
            created.append(instance)
            return instance

        monkeypatch.setattr(
            "app.services.rag_pipeline_builder.VectorStoreFactory.create",
            fake_create,
        )

        pinecone_openai_a = _get_shared_vector_store(
            VectorStoreType.PINECONE,
            EmbeddingProviderType.OPENAI,
        )
        pinecone_openai_b = _get_shared_vector_store(
            VectorStoreType.PINECONE,
            EmbeddingProviderType.OPENAI,
        )
        pinecone_voyage = _get_shared_vector_store(
            VectorStoreType.PINECONE,
            EmbeddingProviderType.VOYAGE,
        )

        assert pinecone_openai_a is pinecone_openai_b
        assert pinecone_openai_a is not pinecone_voyage
        assert len(created) == 2

    def test_build_document_ingestion_service_passes_embedding_provider_to_vector_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _get_shared_vector_store.cache_clear()
        calls: list[tuple[VectorStoreType, EmbeddingProviderType]] = []

        def fake_create(
            store_type: VectorStoreType,
            *,
            embedding_provider: EmbeddingProviderType,
        ) -> object:
            calls.append((store_type, embedding_provider))
            return object()

        monkeypatch.setattr(
            "app.services.rag_pipeline_builder.VectorStoreFactory.create",
            fake_create,
        )

        build_document_ingestion_service(
            RagOptions(
                vector_store=VectorStoreType.PINECONE,
                embedding_provider=EmbeddingProviderType.VOYAGE,
            )
        )

        assert calls == [(VectorStoreType.PINECONE, EmbeddingProviderType.VOYAGE)]
