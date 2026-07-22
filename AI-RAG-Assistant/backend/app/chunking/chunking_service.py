from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk
from app.models.document import ParsedDocument


class ChunkingService:
    """Orchestrates splitting a ParsedDocument into chunks via a configured strategy.

    Parsing, embedding, storage, and retrieval are handled elsewhere. This service
    only delegates to the injected BaseChunker implementation.
    """

    def __init__(self, chunker: BaseChunker) -> None:
        self._chunker : BaseChunker = chunker

    def chunk_document(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        """Return chunks produced by the configured chunker for `parsed_document`."""
        return self._chunker.chunk(parsed_document)
