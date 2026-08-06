import logging

from pydantic import BaseModel

from app.chunking.chunk_models import DocumentChunk
from app.config import CHUNKING_DEBUG, CHUNK_OVERLAP, CHUNK_SIZE
from app.chunking.chunking_service import ChunkingService
from app.embeddings.embedding_models import EmbeddingUsage
from app.embeddings.embedding_service import EmbeddingService
from app.models.document import ParsedDocument
from app.chunking.header_segmentation import segment_by_headers
from app.services.chunking_debug import log_chunking_debug
from app.services.document_exceptions import DocumentIngestionError
from app.services.document_parser import DocumentParser
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorStoreRecord

logger = logging.getLogger(__name__)


class DocumentIndexingResult(BaseModel):
    """Summary of one document indexing run."""

    document_id: str
    filename: str
    indexed_chunk_count: int
    embedding_model: str | None
    usage: EmbeddingUsage | None


class DocumentIngestionService:
    """Coordinates parsing, chunking, embedding, and vector storage for one document."""

    def __init__(
        self,
        document_parser: DocumentParser,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
    ) -> None:
        self._document_parser = document_parser
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_document(
        self,
        document_id: str,
        *,
        chunking_strategy: str | None = None,
    ) -> DocumentIndexingResult:
        """Parse, chunk, embed, and store one uploaded document."""
        logger.info("Parsing document document_id=%s", document_id)
        parsed_document = self._document_parser.parse(document_id)
        _warn_header_strategy_mismatch(parsed_document, chunking_strategy)
        chunks = self._chunking_service.chunk_document(parsed_document)
        logger.info(
            "Generated %d chunks for document_id=%s filename=%s",
            len(chunks),
            parsed_document.document_id,
            parsed_document.filename,
        )

        if not chunks:
            if CHUNKING_DEBUG:
                self._log_chunking_debug(
                    parsed_document=parsed_document,
                    chunks=chunks,
                    records=[],
                    chunking_strategy=chunking_strategy,
                )
            return DocumentIndexingResult(
                document_id=parsed_document.document_id,
                filename=parsed_document.filename,
                indexed_chunk_count=0,
                embedding_model=None,
                usage=None,
            )

        logger.info("Generating embeddings for document_id=%s", document_id)
        batch = self._embedding_service.embed_batch([chunk.text for chunk in chunks])
        if len(batch.embeddings) != len(chunks):
            raise DocumentIngestionError(
                f"Expected {len(chunks)} embeddings for document "
                f"'{document_id}', got {len(batch.embeddings)}"
            )

        records = [
            VectorStoreRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                source_filename=parsed_document.filename,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, batch.embeddings, strict=True)
        ]
        self._vector_store.upsert(records)
        logger.info(
            "Stored %d vectors for document_id=%s filename=%s",
            len(records),
            parsed_document.document_id,
            parsed_document.filename,
        )

        if CHUNKING_DEBUG:
            self._log_chunking_debug(
                parsed_document=parsed_document,
                chunks=chunks,
                records=records,
                chunking_strategy=chunking_strategy,
            )

        return DocumentIndexingResult(
            document_id=parsed_document.document_id,
            filename=parsed_document.filename,
            indexed_chunk_count=len(chunks),
            embedding_model=batch.embeddings[0].model,
            usage=batch.usage,
        )

    def _log_chunking_debug(
        self,
        *,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        records: list[VectorStoreRecord],
        chunking_strategy: str | None,
    ) -> None:
        chunker = getattr(self._chunking_service, "_chunker", None)
        chunk_size = getattr(chunker, "_chunk_size", CHUNK_SIZE)
        chunk_overlap = getattr(chunker, "_chunk_overlap", CHUNK_OVERLAP)
        log_chunking_debug(
            parsed_document=parsed_document,
            chunks=chunks,
            records=records,
            chunking_strategy=chunking_strategy or "unknown",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def _warn_header_strategy_mismatch(
    parsed_document: ParsedDocument,
    chunking_strategy: str | None,
) -> None:
    if chunking_strategy == "header_aware":
        return

    sections = segment_by_headers(parsed_document.extracted_text)
    header_count = sum(1 for header, _ in sections if header)
    if header_count < 2:
        return

    logger.warning(
        "Document '%s' contains %d markdown/html header sections but "
        "chunking_strategy=%s. Enable Advanced RAG and select header_aware "
        "before upload.",
        parsed_document.filename,
        header_count,
        chunking_strategy or "unknown",
    )
