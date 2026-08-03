import uuid

from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk
from app.chunking.table_segmentation import is_table_segment, segment_by_table_blocks
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.models.document import ParsedDocument

# Separator priority from coarsest to finest structural boundaries.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ")
_TABLE_CHUNK_SIZE_MULTIPLIER = 2


class RecursiveChunker(BaseChunker):
    """Recursively split text using progressively finer separators.

    Starts with paragraph boundaries and walks down through lines, sentences,
    words, and finally fixed-size character windows for oversized segments.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        """Split `parsed_document` into ordered chunks."""
        text = parsed_document.extracted_text
        if not text:
            return []

        chunk_texts: list[str] = []
        for segment in segment_by_table_blocks(text):
            is_table = is_table_segment(segment)
            min_separator_index = 1 if is_table else 0
            segment_chunk_size = (
                self._chunk_size * _TABLE_CHUNK_SIZE_MULTIPLIER
                if is_table
                else self._chunk_size
            )
            chunk_texts.extend(
                self._split_text(
                    segment,
                    separator_index=0,
                    min_separator_index=min_separator_index,
                    chunk_size=segment_chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
            )

        return [
            DocumentChunk(
                chunk_id=uuid.uuid4().hex,
                document_id=parsed_document.document_id,
                chunk_index=index,
                text=chunk_text,
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

    def _split_text(
        self,
        text: str,
        separator_index: int = 0,
        min_separator_index: int = 0,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Recursively split `text`, then merge pieces into sized chunks."""
        if separator_index < min_separator_index:
            separator_index = min_separator_index

        if len(text) <= chunk_size:
            return [text]

        if separator_index >= len(_SEPARATORS):
            return self._character_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        separator = _SEPARATORS[separator_index]
        pieces = self._split_with_separator(text, separator)

        if len(pieces) == 1:
            return self._split_text(
                text,
                separator_index + 1,
                min_separator_index=min_separator_index,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        processed: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= chunk_size:
                processed.append(piece)
            else:
                processed.extend(
                    self._split_text(
                        piece,
                        separator_index + 1,
                        min_separator_index=min_separator_index,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )

        return self._merge_pieces(
            processed,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        """Split `text` on `separator` while keeping separators on all but the last piece."""
        if separator not in text:
            return [text]

        parts = text.split(separator)
        if len(parts) == 1:
            return [text]

        segments = [parts[0] + separator]
        segments.extend(part + separator for part in parts[1:-1])
        segments.append(parts[-1])
        return segments

    def _merge_pieces(
        self,
        pieces: list[str],
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Merge adjacent pieces into chunks up to `chunk_size` with overlap."""
        if not pieces:
            return []

        chunks: list[str] = []
        current: list[str] = []
        total = 0

        for piece in pieces:
            piece_len = len(piece)
            if current and total + piece_len > chunk_size:
                chunks.append("".join(current))
                while current and (
                    total > chunk_overlap
                    or (total + piece_len > chunk_size and total > 0)
                ):
                    total -= len(current[0])
                    current = current[1:]

            current.append(piece)
            total += piece_len

        if current:
            chunks.append("".join(current))

        return chunks

    def _character_chunk(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Fixed-size fallback when no separator can reduce a segment further."""
        if len(text) <= chunk_size:
            return [text]

        step = chunk_size - chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if not chunk_text:
                break

            chunks.append(chunk_text)
            if end >= len(text):
                break
            start += step

        return chunks
