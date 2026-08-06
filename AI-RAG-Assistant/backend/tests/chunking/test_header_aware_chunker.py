import pytest

from app.chunking.header_aware_chunker import HeaderAwareChunker
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str, document_id: str = DOCUMENT_ID) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="sample.md",
        extracted_text=text,
    )


class TestHeaderAwareChunkerValidation:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(0, 0), (-1, 0)],
    )
    def test_rejects_non_positive_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            HeaderAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_rejects_negative_chunk_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            HeaderAwareChunker(chunk_size=100, chunk_overlap=-1)

    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(10, 10), (10, 11)],
    )
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            HeaderAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestHeaderAwareChunkerChunking:
    def test_empty_document_returns_empty_list(self) -> None:
        chunker = HeaderAwareChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk(_parsed("")) == []

    def test_plain_text_without_headers_falls_back_to_recursive_splitting(self) -> None:
        first_paragraph = "Alpha text here."
        second_paragraph = "Beta text here."
        text = f"{first_paragraph}\n\n{second_paragraph}"
        chunker = HeaderAwareChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert chunks[0].text == f"{first_paragraph}\n\n"
        assert chunks[1].text == second_paragraph

    def test_keeps_small_section_under_one_header_together(self) -> None:
        text = "## Policies\n\nRefunds are available within 30 days."
        chunker = HeaderAwareChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_multiple_header_levels_produce_separate_chunks(self) -> None:
        text = (
            "# Overview\n\n"
            "Intro paragraph.\n\n"
            "## Details\n\n"
            "More detail here."
        )
        chunker = HeaderAwareChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert chunks[0].text.startswith("# Overview\n\nIntro paragraph.")
        assert chunks[1].text.startswith("## Details\n\nMore detail here.")

    def test_preserves_header_in_every_chunk_when_section_is_oversized(self) -> None:
        body = "Sentence one. " * 12
        text = f"## Long Section\n\n{body.strip()}"
        chunker = HeaderAwareChunker(chunk_size=60, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) > 1
        assert all(chunk.text.startswith("## Long Section\n\n") for chunk in chunks)
        assert all(len(chunk.text) <= 60 for chunk in chunks)

    def test_preamble_before_first_header_is_chunked_without_header_prefix(self) -> None:
        text = "Preamble paragraph.\n\n# Title\n\nBody paragraph."
        chunker = HeaderAwareChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert chunks[0].text == "Preamble paragraph."
        assert chunks[1].text == "# Title\n\nBody paragraph."

    def test_html_headings_are_detected(self) -> None:
        text = "<h2>Summary</h2>\n\nKey points here."
        chunker = HeaderAwareChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == "<h2>Summary</h2>\n\nKey points here."

    def test_very_large_paragraph_in_headerless_document_is_split(self) -> None:
        text = "Paragraph start. " + ("detail " * 30)
        chunker = HeaderAwareChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 50 for chunk in chunks)

    def test_chunk_indexes_are_sequential(self) -> None:
        text = "# One\n\nAlpha.\n\n# Two\n\nBeta."
        chunker = HeaderAwareChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_document_id_is_preserved(self) -> None:
        text = "## Section\n\nBody text."
        chunker = HeaderAwareChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text, document_id=DOCUMENT_ID))

        assert chunks
        assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        text = "## Section\n\n" + ("Body sentence. " * 10)
        chunker = HeaderAwareChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
