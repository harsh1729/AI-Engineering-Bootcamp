"""Temporary structured logging for chunking/indexing pipeline inspection."""

import logging

from app.chunking.chunk_models import DocumentChunk
from app.models.document import ParsedDocument
from app.vector_store.vector_store_models import VectorStoreRecord

logger = logging.getLogger(__name__)

_HEADER = "=" * 58
_CHUNK_SEP = "-" * 58


def compute_chunk_offsets(
    full_text: str, chunks: list[DocumentChunk]
) -> list[tuple[int, int]]:
    """Locate each chunk in `full_text`; offsets are best-effort when text repeats."""
    search_from = 0
    offsets: list[tuple[int, int]] = []
    for chunk in chunks:
        pos = full_text.find(chunk.text, search_from)
        if pos == -1:
            pos = full_text.find(chunk.text)
        if pos == -1:
            pos = search_from
        offsets.append((pos, pos + len(chunk.text)))
        search_from = max(search_from, pos + 1)
    return offsets


def log_chunking_debug(
    *,
    parsed_document: ParsedDocument,
    chunks: list[DocumentChunk],
    records: list[VectorStoreRecord],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Emit a structured chunking debug block. Caller should gate on CHUNKING_DEBUG."""
    lines = [
        _HEADER,
        "CHUNKING DEBUG",
        _HEADER,
        "",
        "Document:",
        parsed_document.filename,
        "Chunking Strategy:",
        chunking_strategy,
        "Chunk Size:",
        str(chunk_size),
        "Chunk Overlap:",
        str(chunk_overlap),
        "Total Chunks Created:",
        str(len(chunks)),
        "",
    ]

    offsets = compute_chunk_offsets(parsed_document.extracted_text, chunks)
    for chunk, (start, end) in zip(chunks, offsets, strict=True):
        lines.extend(
            [
                _CHUNK_SEP,
                f"Chunk #{chunk.chunk_index}",
                "Characters:",
                str(len(chunk.text)),
                "Start Offset:",
                str(start),
                "End Offset:",
                str(end),
                "Preview:",
                chunk.text,
                "",
            ]
        )

    vector_ids = [record.chunk_id for record in records]
    mapping_lines = [
        f"  {record.chunk_index} -> {record.chunk_id}" for record in records
    ]

    lines.extend(
        [
            "Indexed chunk count:",
            str(len(records)),
            "Vector IDs:",
            ", ".join(vector_ids) if vector_ids else "(none)",
            "Chunk Index -> Vector ID mapping",
            *(mapping_lines if mapping_lines else ["  (none)"]),
            _HEADER,
            "",
        ]
    )

    logger.info("\n".join(lines))
