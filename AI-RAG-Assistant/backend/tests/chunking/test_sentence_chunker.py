import pytest

from app.chunking.sentence_chunker import SentenceChunker
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str, document_id: str = DOCUMENT_ID) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="sample.txt",
        extracted_text=text,
    )


class TestSentenceChunkerValidation:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(0, 0), (-1, 0)],
    )
    def test_rejects_non_positive_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_rejects_negative_chunk_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            SentenceChunker(chunk_size=100, chunk_overlap=-1)

    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(10, 10), (10, 11)],
    )
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestSentenceChunkerChunking:
    def test_empty_document_returns_empty_list(self) -> None:
        chunker = SentenceChunker(chunk_size=50, chunk_overlap=0)
        assert chunker.chunk(_parsed("")) == []

    def test_keeps_short_sentences_intact(self) -> None:
        text = "First sentence. Second sentence."
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_splits_on_sentence_boundaries_before_size_limit(self) -> None:
        text = "Alpha sentence here. Beta sentence here."
        chunker = SentenceChunker(chunk_size=25, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert chunks[0].text == "Alpha sentence here. "
        assert chunks[1].text == "Beta sentence here."

    def test_single_extremely_long_sentence_uses_character_fallback(self) -> None:
        text = "Word " + ("x" * 40)
        chunker = SentenceChunker(chunk_size=15, chunk_overlap=3)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) >= 2
        assert all(len(chunk.text) <= 15 for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks).startswith("Word x")

    def test_very_large_paragraph_splits_into_multiple_chunks(self) -> None:
        text = "Sentence one. " * 20
        chunker = SentenceChunker(chunk_size=40, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 40 for chunk in chunks)

    def test_overlap_applies_when_sentence_exceeds_chunk_size(self) -> None:
        text = "A" * 25
        chunker = SentenceChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 3
        assert chunks[0].text == "A" * 10
        assert chunks[1].text == "A" * 10
        assert chunks[0].text[-2:] == chunks[1].text[:2]

    def test_chunk_indexes_are_sequential(self) -> None:
        text = "One. Two. Three. Four. Five."
        chunker = SentenceChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_document_id_is_preserved(self) -> None:
        text = "Sample sentence."
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text, document_id=DOCUMENT_ID))

        assert chunks
        assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        text = "Repeat. " * 10
        chunker = SentenceChunker(chunk_size=12, chunk_overlap=0)
        chunks = chunker.chunk(_parsed(text))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
