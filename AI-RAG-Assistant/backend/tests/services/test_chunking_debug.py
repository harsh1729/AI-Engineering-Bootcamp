import logging

from app.chunking.chunk_models import DocumentChunk
from app.embeddings.embedding_models import EmbeddingVector
from app.models.document import ParsedDocument
from app.services.chunking_debug import compute_chunk_offsets, log_chunking_debug
from app.vector_store.vector_store_models import VectorStoreRecord

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class TestChunkingDebug:
    def test_compute_chunk_offsets_finds_sequential_positions(self) -> None:
        full_text = "Alpha segment. Beta segment."
        chunks = [
            DocumentChunk(
                chunk_id="chunk-0",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                text="Alpha segment.",
            ),
            DocumentChunk(
                chunk_id="chunk-1",
                document_id=DOCUMENT_ID,
                chunk_index=1,
                text="Beta segment.",
            ),
        ]

        assert compute_chunk_offsets(full_text, chunks) == [(0, 14), (15, 28)]

    def test_logs_full_chunking_debug_block(self, caplog) -> None:
        parsed = ParsedDocument(
            document_id=DOCUMENT_ID,
            filename="fixture.txt",
            extracted_text="First chunk body. Second chunk body.",
        )
        chunks = [
            DocumentChunk(
                chunk_id="vec-0",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                text="First chunk body.",
            ),
            DocumentChunk(
                chunk_id="vec-1",
                document_id=DOCUMENT_ID,
                chunk_index=1,
                text="Second chunk body.",
            ),
        ]
        records = [
            VectorStoreRecord(
                chunk_id="vec-0",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                text=chunks[0].text,
                source_filename="fixture.txt",
                embedding=EmbeddingVector(embedding=[0.1], model="text-embedding-3-small"),
            ),
            VectorStoreRecord(
                chunk_id="vec-1",
                document_id=DOCUMENT_ID,
                chunk_index=1,
                text=chunks[1].text,
                source_filename="fixture.txt",
                embedding=EmbeddingVector(embedding=[0.2], model="text-embedding-3-small"),
            ),
        ]

        with caplog.at_level(logging.INFO, logger="app.services.chunking_debug"):
            log_chunking_debug(
                parsed_document=parsed,
                chunks=chunks,
                records=records,
                chunking_strategy="sentence",
                chunk_size=1500,
                chunk_overlap=300,
            )

        output = caplog.text
        assert "CHUNKING DEBUG" in output
        assert "fixture.txt" in output
        assert "sentence" in output
        assert "Chunk #0" in output
        assert "First chunk body." in output
        assert "Chunk #1" in output
        assert "Second chunk body." in output
        assert "Indexed chunk count:" in output
        assert "Vector IDs:" in output
        assert "vec-0, vec-1" in output
        assert "0 -> vec-0" in output
        assert "1 -> vec-1" in output
