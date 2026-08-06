import re
import uuid

from app.chunking.abbreviations import COMMON_ABBREVIATIONS
from app.chunking.chunk_models import DocumentChunk

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+\s+")


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap configuration."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")


def split_into_sentences(text: str) -> list[str]:
    """Split `text` into sentence-like segments while keeping trailing punctuation."""
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if _is_abbreviation_boundary(text, match):
            continue

        sentences.append(text[start : match.end()])
        start = match.end()

    remainder = text[start:]
    if remainder:
        sentences.append(remainder)

    return sentences


def _is_abbreviation_boundary(text: str, match: re.Match[str]) -> bool:
    """Return True when the matched punctuation ends a known abbreviation."""
    if not match.group().startswith("."):
        return False

    token = _token_before_period(text, match.start())
    return token in COMMON_ABBREVIATIONS


def _token_before_period(text: str, period_index: int) -> str:
    """Return the normalized word/token immediately before a period."""
    end = period_index
    start = period_index - 1
    while start >= 0 and (text[start].isalnum() or text[start] in ".-"):
        start -= 1
    return text[start + 1 : end].replace(".", "").lower()


def merge_text_pieces(
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


def split_by_character_window(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split `text` into fixed-size windows when no finer boundary applies."""
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


def build_document_chunks(document_id: str, chunk_texts: list[str]) -> list[DocumentChunk]:
    """Wrap chunk text segments in `DocumentChunk` models with stable ordering."""
    return [
        DocumentChunk(
            chunk_id=uuid.uuid4().hex,
            document_id=document_id,
            chunk_index=index,
            text=chunk_text,
        )
        for index, chunk_text in enumerate(chunk_texts)
    ]
