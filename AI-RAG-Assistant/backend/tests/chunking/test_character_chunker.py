import pytest

from app.chunking.character_chunker import CharacterChunker
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str, document_id: str = DOCUMENT_ID) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="sample.txt",
        extracted_text=text,
    )


class TestCharacterChunkerValidation:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [
            (0, 0),
            (-1, 0),
        ],
    )
    def test_rejects_non_positive_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            CharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_rejects_negative_chunk_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            CharacterChunker(chunk_size=100, chunk_overlap=-1)

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
            CharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestCharacterChunkerChunking:
    def test_empty_document_returns_empty_list(self) -> None:
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        assert chunker.chunk(_parsed("")) == []

    def test_document_smaller_than_chunk_size_returns_one_chunk(self) -> None:
        text = "hello"
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_document_exactly_equal_to_chunk_size_returns_one_chunk(self) -> None:
        text = "0123456789"
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_document_larger_than_chunk_size_produces_multiple_chunks(self) -> None:
        text = "0123456789abcdefghij"
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) > 1
        assert chunks[0].text == text[:10]
        assert chunks[1].text == text[8:18]

    def test_overlap_starts_second_chunk_at_correct_index(self) -> None:
        text = "0123456789abcdefghij"
        chunk_size = 10
        chunk_overlap = 2
        chunker = CharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) >= 2
        assert chunks[0].text == text[0:chunk_size]
        # step = chunk_size - chunk_overlap = 8
        assert chunks[1].text == text[8 : 8 + chunk_size]
        assert text[8:10] == chunks[0].text[-2:]
        assert text[8:10] == chunks[1].text[:2]

    def test_final_chunk_can_be_shorter_than_chunk_size(self) -> None:
        text = "0123456789abcde"  # length 15
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert len(chunks) == 2
        assert len(chunks[0].text) == 10
        assert len(chunks[1].text) == 7
        assert chunks[1].text == text[8:]

    def test_chunk_indexes_are_sequential_from_zero(self) -> None:
        text = "0123456789abcdefghijklmnop"
        chunker = CharacterChunker(chunk_size=8, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_every_chunk_has_correct_document_id(self) -> None:
        text = "0123456789abcdefghij"
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text, document_id=DOCUMENT_ID))

        assert chunks
        assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)

    def test_every_chunk_has_unique_chunk_id(self) -> None:
        text = "0123456789abcdefghijklmnop"
        chunker = CharacterChunker(chunk_size=6, chunk_overlap=2)
        chunks = chunker.chunk(_parsed(text))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert all(len(chunk_id) == 32 for chunk_id in chunk_ids)
