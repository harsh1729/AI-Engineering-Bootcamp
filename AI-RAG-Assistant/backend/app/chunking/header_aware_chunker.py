from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk
from app.chunking.chunk_size_utils import (
    build_document_chunks,
    split_by_character_window,
    validate_chunk_params,
)
from app.chunking.header_segmentation import segment_by_headers
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.models.document import ParsedDocument


class HeaderAwareChunker(BaseChunker):
    """Split documents by detected heading sections before applying size limits.

    Heading detection is delegated to `header_segmentation` so format-specific
    parsers can inject normalized markers (DOCX styles, PDF layout heuristics)
    or add new detectors without changing chunking logic.
    """

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

        sections = segment_by_headers(text)
        if not sections:
            return RecursiveChunker(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            ).chunk(parsed_document)

        chunk_texts: list[str] = []
        for header, body in sections:
            chunk_texts.extend(self._chunk_section(header, body, parsed_document))

        return build_document_chunks(parsed_document.document_id, chunk_texts)

    def _chunk_section(
        self,
        header: str,
        body: str,
        parsed_document: ParsedDocument,
    ) -> list[str]:
        if not header:
            return self._split_plain_text(body, parsed_document)

        if body:
            section_text = f"{header}\n\n{body}"
            prefix = f"{header}\n\n"
        else:
            section_text = header
            prefix = f"{header}\n"

        if len(section_text) <= self._chunk_size:
            return [section_text]

        body_budget = self._chunk_size - len(prefix)
        if body_budget <= 0:
            return split_by_character_window(
                section_text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

        body_chunks = self._split_plain_text(body, parsed_document, body_budget)
        return [f"{prefix}{chunk_text}" for chunk_text in body_chunks]

    def _split_plain_text(
        self,
        text: str,
        parsed_document: ParsedDocument,
        chunk_size: int | None = None,
    ) -> list[str]:
        if not text:
            return []

        sub_document = ParsedDocument(
            document_id=parsed_document.document_id,
            filename=parsed_document.filename,
            extracted_text=text,
        )
        chunks = SentenceChunker(
            chunk_size=chunk_size or self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        ).chunk(sub_document)
        return [chunk.text for chunk in chunks]
