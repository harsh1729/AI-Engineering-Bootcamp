from pydantic import BaseModel

from app.chunking.chunking_service import ChunkingService
from app.embeddings.embedding_models import EmbeddingUsage
from app.embeddings.embedding_service import EmbeddingService
from app.models.document import ParsedDocument
from app.services.document_exceptions import DocumentIngestionError
from app.services.document_parser import DocumentParser
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorStoreRecord


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

    def index_document(self, document_id: str) -> DocumentIndexingResult:
        """Parse, chunk, embed, and store one uploaded document."""
        parsed_document = self._document_parser.parse(document_id)
        chunks = self._chunking_service.chunk_document(parsed_document)

        if not chunks:
            return DocumentIndexingResult(
                document_id=parsed_document.document_id,
                filename=parsed_document.filename,
                indexed_chunk_count=0,
                embedding_model=None,
                usage=None,
            )

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

        return DocumentIndexingResult(
            document_id=parsed_document.document_id,
            filename=parsed_document.filename,
            indexed_chunk_count=len(chunks),
            embedding_model=batch.embeddings[0].model,
            usage=batch.usage,
        )
