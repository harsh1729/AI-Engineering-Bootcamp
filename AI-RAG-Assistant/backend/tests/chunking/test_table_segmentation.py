import pytest

from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.table_segmentation import is_table_segment, segment_by_table_blocks
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str) -> ParsedDocument:
    return ParsedDocument(
        document_id=DOCUMENT_ID,
        filename="sample.txt",
        extracted_text=text,
    )


class TestTableSegmentation:
    def test_leaves_non_table_text_as_single_segment(self) -> None:
        text = "Intro paragraph.\n\nAnother paragraph."
        assert segment_by_table_blocks(text) == [text]

    def test_starts_new_segment_at_table_block(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            "TABLE 7: IBRD TOP COUNTRY BORROWERS, FISCAL 2025\n"
            "Brazil 3,856"
        )
        segments = segment_by_table_blocks(text)

        assert len(segments) == 2
        assert segments[0] == "Intro paragraph."
        assert segments[1].startswith("\n\nTABLE 7:")
        assert is_table_segment(segments[1])

    def test_is_table_segment_false_for_regular_text(self) -> None:
        assert not is_table_segment("Regular paragraph text.")


class TestRecursiveChunkerTableAwareness:
    def test_keeps_table_heading_with_following_rows(self) -> None:
        text = (
            "Preamble text.\n\n"
            "TABLE 7: IBRD TOP COUNTRY BORROWERS, FISCAL 2025\n"
            "MILLIONS OF DOLLARS\n\n"
            " COUNTRY                        COMMITMENTS\n"
            " Brazil                                     3,856\n"
            " Türkiye                                         3,791\n"
            " Argentina                                   3,730\n"
            "\n\n"
            "IBRD FINANCIAL RESOURCES AND FINANCIAL MODEL\n"
            "Narrative section continues here."
        )
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk(_parsed(text))

        table_chunks = [
            chunk
            for chunk in chunks
            if "TABLE 7" in chunk.text and "TOP COUNTRY BORROWERS" in chunk.text
        ]
        assert table_chunks, "Expected a chunk containing the TABLE 7 heading"

        heading_chunk = table_chunks[0]
        assert "Brazil" in heading_chunk.text
        assert "3,856" in heading_chunk.text
        assert "Argentina" in heading_chunk.text

    def test_splits_non_table_text_on_paragraphs(self) -> None:
        first_paragraph = "Alpha text here."
        second_paragraph = "Beta text here."
        text = f"{first_paragraph}\n\n{second_paragraph}"
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert chunks[0].text == f"{first_paragraph}\n\n"
        assert chunks[1].text == second_paragraph

    def test_table_segment_uses_double_chunk_size(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            "TABLE 7: IBRD TOP COUNTRY BORROWERS, FISCAL 2025\n"
            + ("Row data line with values.\n" * 200)
        )
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(_parsed(text))

        table_chunk = next(chunk for chunk in chunks if "TABLE 7" in chunk.text)
        assert len(table_chunk.text) > 100
        assert len(table_chunk.text) <= 200

    @pytest.mark.parametrize(
        "table_text",
        [
            pytest.param(
                (
                    "TABLE 5: IBRD COMMITMENTS BY SECTOR, FISCAL 2025\n"
                    "MILLIONS OF DOLLARS\n\n"
                    " Agriculture, Fishing, and Forestry 2,905\n"
                    " Digital Development 782\n"
                ),
                id="sector-table",
            ),
        ],
    )
    def test_table_segment_skips_paragraph_separator(self, table_text: str) -> None:
        text = f"Lead-in paragraph.\n\n{table_text}\n\nAfter table narrative."
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk(_parsed(text))

        table_chunk = next(chunk for chunk in chunks if "TABLE 5" in chunk.text)
        assert "Agriculture, Fishing, and Forestry" in table_chunk.text
        assert "2,905" in table_chunk.text
