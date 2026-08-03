from app.models.rag_config import ChunkingStrategy, RagOptions
from app.services.rag_pipeline_builder import (
    _get_shared_chunking_service,
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
