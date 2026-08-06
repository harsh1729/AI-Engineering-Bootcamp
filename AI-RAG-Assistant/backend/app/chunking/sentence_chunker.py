import uuid

from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk
from app.chunking.chunk_size_utils import (
    build_document_chunks,
    split_by_character_window,
    split_into_sentences,
    validate_chunk_params,
    merge_text_pieces,
)
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.models.document import ParsedDocument


class SentenceChunker(BaseChunker):
    """Split text on sentence boundaries, merging sentences up to `chunk_size`."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        validate_chunk_params(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        text = parsed_document.extracted_text
        if not text:
            return []

        pieces: list[str] = []
        for sentence in split_into_sentences(text):
            if len(sentence) <= self._chunk_size:
                pieces.append(sentence)
            else:
                pieces.extend(
                    split_by_character_window(
                        sentence,
                        chunk_size=self._chunk_size,
                        chunk_overlap=self._chunk_overlap,
                    )
                )

        chunk_texts = merge_text_pieces(
            pieces,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return build_document_chunks(parsed_document.document_id, chunk_texts)
