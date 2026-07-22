import pytest

from app.chunking.recursive_chunker import RecursiveChunker
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str, document_id: str = DOCUMENT_ID) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="sample.txt",
        extracted_text=text,
    )


def _reconstruct_without_overlap(chunks: list) -> str:
    return "".join(chunk.text for chunk in chunks)


class TestRecursiveChunkerValidation:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [
            (0, 0),
            (-1, 0),
        ],
    )
    def test_rejects_non_positive_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_rejects_negative_chunk_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            RecursiveChunker(chunk_size=100, chunk_overlap=-1)

    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [
            (10, 10),
            (10, 11),
        ],
    )
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestRecursiveChunkerChunking:
    def test_empty_document_returns_empty_list(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        assert chunker.chunk(_parsed("")) == []

    def test_small_document_produces_one_chunk(self) -> None:
        text = "Short document text."
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_paragraph_splitting_occurs_before_sentence_splitting(self) -> None:
        first_paragraph = "Alpha text here."
        second_paragraph = "Beta text here."
        text = f"{first_paragraph}\n\n{second_paragraph}"
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert chunks[0].text == f"{first_paragraph}\n\n"
        assert chunks[1].text == second_paragraph
        assert _reconstruct_without_overlap(chunks) == text

    def test_sentence_splitting_occurs_before_word_splitting(self) -> None:
        text = "Hello there world. More words here."
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) >= 2
        assert chunks[0].text == "Hello there world. "
        assert chunks[1].text == "More words here."

    def test_word_splitting_occurs_before_character_fallback(self) -> None:
        long_word = "abcdefghijklmnop"
        text = f"short {long_word} end"
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        chunk_texts = [chunk.text for chunk in chunks]
        assert any(chunk_text == "short " for chunk_text in chunk_texts)
        assert any(long_word[:10] in chunk_text for chunk_text in chunk_texts)
        assert _reconstruct_without_overlap(chunks) == text

    def test_very_long_word_falls_back_to_character_chunking(self) -> None:
        text = "a" * 25
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 3
        assert all(len(chunk.text) <= 10 for chunk in chunks)
        assert chunks[0].text == text[0:10]
        assert chunks[1].text == text[8:18]
        assert chunks[2].text == text[16:25]

    def test_no_text_is_lost_after_chunking(self) -> None:
        text = (
            "Intro paragraph with detail.\n\n"
            "Second paragraph has multiple sentences. Here is another one!\n"
            "And a final question? "
            "supercalifragilisticexpialidocious"
        )
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert chunks
        assert _reconstruct_without_overlap(chunks) == text

    def test_chunk_indexes_are_sequential(self) -> None:
        text = "One. Two. Three. Four. Five. Six."
        chunker = RecursiveChunker(chunk_size=12, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_document_id_is_preserved(self) -> None:
        text = "Sample text for document id check."
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text, document_id=DOCUMENT_ID))

        assert chunks
        assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        text = "Word " * 20
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert all(len(chunk_id) == 32 for chunk_id in chunk_ids)
