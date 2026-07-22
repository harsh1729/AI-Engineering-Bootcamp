import uuid

from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk
from app.models.document import ParsedDocument


class CharacterChunker(BaseChunker):
    """Sliding-window chunker that splits text by fixed character counts."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        text = parsed_document.extracted_text
        if not text:
            return []

        step = self._chunk_size - self._chunk_overlap
        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]
            if not chunk_text:
                break

            chunks.append(
                DocumentChunk(
                    chunk_id=uuid.uuid4().hex,
                    document_id=parsed_document.document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1
            if end >= len(text):
                break
            start += step

        return chunks
