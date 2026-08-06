from unittest.mock import MagicMock, create_autospec

import pytest

from app.chunking.chunk_models import DocumentChunk
from app.chunking.chunking_service import ChunkingService
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingUsage, EmbeddingVector
from app.embeddings.embedding_service import EmbeddingService
from app.models.document import ParsedDocument
from app.services.document_exceptions import DocumentIngestionError, DocumentParsingError
from app.services.document_ingestion import DocumentIngestionService, DocumentIndexingResult
from app.services.document_parser import DocumentParser
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorStoreRecord

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"
MODEL = "text-embedding-3-small"


def _parsed(text: str = "First chunk text. Second chunk text.") -> ParsedDocument:
    return ParsedDocument(
        document_id=DOCUMENT_ID,
        filename="handbook.pdf",
        extracted_text=text,
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-0",
            document_id=DOCUMENT_ID,
            chunk_index=0,
            text="First chunk text.",
        ),
        DocumentChunk(
            chunk_id="chunk-1",
            document_id=DOCUMENT_ID,
            chunk_index=1,
            text="Second chunk text.",
        ),
    ]


def _batch_response(
    *,
    embeddings: list[list[float]] | None = None,
) -> EmbeddingBatchResponse:
    vectors = embeddings or [[0.1, 0.2], [0.3, 0.4]]
    return EmbeddingBatchResponse(
        embeddings=[
            EmbeddingVector(embedding=vector, model=MODEL) for vector in vectors
        ],
        usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
    )


@pytest.fixture
def document_parser() -> MagicMock:
    return create_autospec(DocumentParser, instance=True)


@pytest.fixture
def chunking_service() -> MagicMock:
    return create_autospec(ChunkingService, instance=True)


@pytest.fixture
def embedding_service() -> MagicMock:
    return create_autospec(EmbeddingService, instance=True)


@pytest.fixture
def vector_store() -> MagicMock:
    return create_autospec(BaseVectorStore, instance=True)


@pytest.fixture
def ingestion_service(
    document_parser: MagicMock,
    chunking_service: MagicMock,
    embedding_service: MagicMock,
    vector_store: MagicMock,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        document_parser=document_parser,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


class TestDocumentIngestionServiceIndexDocument:
    def test_indexes_document_with_one_embed_batch_and_one_upsert(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        parsed = _parsed()
        chunks = _chunks()
        batch = _batch_response()
        document_parser.parse.return_value = parsed
        chunking_service.chunk_document.return_value = chunks
        embedding_service.embed_batch.return_value = batch

        result = ingestion_service.index_document(DOCUMENT_ID)

        document_parser.parse.assert_called_once_with(DOCUMENT_ID)
        chunking_service.chunk_document.assert_called_once_with(parsed)
        embedding_service.embed_batch.assert_called_once_with(
            ["First chunk text.", "Second chunk text."]
        )
        vector_store.upsert.assert_called_once()
        upserted_records = vector_store.upsert.call_args.args[0]
        assert upserted_records == [
            VectorStoreRecord(
                chunk_id="chunk-0",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                text="First chunk text.",
                source_filename="handbook.pdf",
                embedding=batch.embeddings[0],
            ),
            VectorStoreRecord(
                chunk_id="chunk-1",
                document_id=DOCUMENT_ID,
                chunk_index=1,
                text="Second chunk text.",
                source_filename="handbook.pdf",
                embedding=batch.embeddings[1],
            ),
        ]
        assert result == DocumentIndexingResult(
            document_id=DOCUMENT_ID,
            filename="handbook.pdf",
            indexed_chunk_count=2,
            embedding_model=MODEL,
            usage=batch.usage,
        )

    def test_pairs_chunks_with_embeddings_by_index(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        chunks = _chunks()
        batch = _batch_response(embeddings=[[1.0, 2.0], [3.0, 4.0]])
        document_parser.parse.return_value = _parsed()
        chunking_service.chunk_document.return_value = chunks
        embedding_service.embed_batch.return_value = batch

        ingestion_service.index_document(DOCUMENT_ID)

        first, second = vector_store.upsert.call_args.args[0]
        assert first.embedding.embedding == [1.0, 2.0]
        assert first.embedding.model == MODEL
        assert second.embedding.embedding == [3.0, 4.0]
        assert second.embedding.model == MODEL

    def test_maps_parsed_document_filename_into_every_vector_store_record(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        document_parser.parse.return_value = _parsed()
        chunking_service.chunk_document.return_value = _chunks()
        embedding_service.embed_batch.return_value = _batch_response()

        ingestion_service.index_document(DOCUMENT_ID)

        records = vector_store.upsert.call_args.args[0]
        assert all(record.source_filename == "handbook.pdf" for record in records)

    def test_empty_document_skips_embedding_and_storage(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        parsed = _parsed(text="")
        document_parser.parse.return_value = parsed
        chunking_service.chunk_document.return_value = []

        result = ingestion_service.index_document(DOCUMENT_ID)

        embedding_service.embed_batch.assert_not_called()
        vector_store.upsert.assert_not_called()
        assert result == DocumentIndexingResult(
            document_id=DOCUMENT_ID,
            filename="handbook.pdf",
            indexed_chunk_count=0,
            embedding_model=None,
            usage=None,
        )

    def test_embedding_count_mismatch_raises_and_does_not_upsert(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        document_parser.parse.return_value = _parsed()
        chunking_service.chunk_document.return_value = _chunks()
        embedding_service.embed_batch.return_value = _batch_response(
            embeddings=[[0.1, 0.2]]
        )

        with pytest.raises(DocumentIngestionError, match="Expected 2 embeddings"):
            ingestion_service.index_document(DOCUMENT_ID)

        vector_store.upsert.assert_not_called()

    def test_parser_errors_propagate_unchanged(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        document_parser.parse.side_effect = DocumentParsingError("parse failed")

        with pytest.raises(DocumentParsingError, match="parse failed"):
            ingestion_service.index_document(DOCUMENT_ID)

        chunking_service.chunk_document.assert_not_called()
        embedding_service.embed_batch.assert_not_called()
        vector_store.upsert.assert_not_called()

    def test_emits_chunking_debug_when_flag_enabled(
        self,
        ingestion_service: DocumentIngestionService,
        document_parser: MagicMock,
        chunking_service: MagicMock,
        embedding_service: MagicMock,
        vector_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ) -> None:
        import logging

        monkeypatch.setattr("app.services.document_ingestion.CHUNKING_DEBUG", True)
        document_parser.parse.return_value = _parsed()
        chunking_service.chunk_document.return_value = _chunks()
        mock_chunker = MagicMock()
        mock_chunker._chunk_size = 1500
        mock_chunker._chunk_overlap = 300
        chunking_service._chunker = mock_chunker
        embedding_service.embed_batch.return_value = _batch_response()

        with caplog.at_level(logging.INFO):
            ingestion_service.index_document(
                DOCUMENT_ID,
                chunking_strategy="sentence",
            )

        assert "CHUNKING DEBUG" in caplog.text
        assert "sentence" in caplog.text
        assert "First chunk text." in caplog.text
        assert "chunk-0" in caplog.text

