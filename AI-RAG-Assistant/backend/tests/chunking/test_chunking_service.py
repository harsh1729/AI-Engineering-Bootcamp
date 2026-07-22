from unittest.mock import patch

from app.chunking.character_chunker import CharacterChunker
from app.chunking.chunking_service import ChunkingService
from app.chunking.recursive_chunker import RecursiveChunker
from app.models.document import ParsedDocument

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _parsed(text: str, document_id: str = DOCUMENT_ID) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename="sample.txt",
        extracted_text=text,
    )


def _chunk_fields(chunks: list) -> list[tuple[str, int, str]]:
    return [(chunk.document_id, chunk.chunk_index, chunk.text) for chunk in chunks]


class TestChunkingServiceDelegation:
    def test_delegates_to_character_chunker(self) -> None:
        text = "0123456789abcdefghij"
        parsed = _parsed(text)
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        service = ChunkingService(chunker)

        chunks = service.chunk_document(parsed)

        assert _chunk_fields(chunks) == _chunk_fields(
            CharacterChunker(chunk_size=10, chunk_overlap=2).chunk(parsed)
        )

    def test_delegates_to_recursive_chunker(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph here."
        parsed = _parsed(text)
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        service = ChunkingService(chunker)

        chunks = service.chunk_document(parsed)

        assert _chunk_fields(chunks) == _chunk_fields(
            RecursiveChunker(chunk_size=20, chunk_overlap=0).chunk(parsed)
        )

    def test_returns_chunks_unchanged_from_configured_chunker(self) -> None:
        text = "Hello there world. More words here."
        parsed = _parsed(text)
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        expected = chunker.chunk(parsed)
        service = ChunkingService(chunker)

        with patch.object(chunker, "chunk", return_value=expected) as mock_chunk:
            chunks = service.chunk_document(parsed)

        mock_chunk.assert_called_once_with(parsed)
        assert chunks is expected


class TestChunkingServiceDependencyInjection:
    def test_uses_injected_character_chunker(self) -> None:
        text = "0123456789abcdefghij"
        parsed = _parsed(text)
        character_chunker = CharacterChunker(chunk_size=10, chunk_overlap=2)
        recursive_chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
        character_service = ChunkingService(character_chunker)
        recursive_service = ChunkingService(recursive_chunker)

        assert character_service._chunker is character_chunker
        assert recursive_service._chunker is recursive_chunker

        character_chunks = character_service.chunk_document(parsed)
        recursive_chunks = recursive_service.chunk_document(parsed)

        assert _chunk_fields(character_chunks) == _chunk_fields(
            CharacterChunker(chunk_size=10, chunk_overlap=2).chunk(parsed)
        )
        assert _chunk_fields(recursive_chunks) == _chunk_fields(
            RecursiveChunker(chunk_size=10, chunk_overlap=2).chunk(parsed)
        )
        assert character_chunks != recursive_chunks

    def test_swapping_chunker_changes_behavior(self) -> None:
        text = "Alpha text here.\n\nBeta text here."
        parsed = _parsed(text)
        character_service = ChunkingService(CharacterChunker(chunk_size=20, chunk_overlap=0))
        recursive_service = ChunkingService(RecursiveChunker(chunk_size=20, chunk_overlap=0))

        character_chunks = character_service.chunk_document(parsed)
        recursive_chunks = recursive_service.chunk_document(parsed)

        assert len(recursive_chunks) == 2
        assert recursive_chunks[0].text == "Alpha text here.\n\n"
        assert recursive_chunks[1].text == "Beta text here."
        assert character_chunks[0].text == text[:20]
        assert _chunk_fields(character_chunks) != _chunk_fields(recursive_chunks)
