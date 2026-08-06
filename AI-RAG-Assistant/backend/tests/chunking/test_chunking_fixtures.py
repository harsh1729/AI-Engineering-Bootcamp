from pathlib import Path

import pytest

from app.chunking.header_aware_chunker import HeaderAwareChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.config import CHUNK_SIZE
from app.models.document import ParsedDocument

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "chunking"
SENTENCE_FIXTURE = FIXTURES_DIR / "Sentence_Chunker_Test.txt"
HEADER_AWARE_FIXTURE = FIXTURES_DIR / "Markdown_Header_Chunker_Test.txt"

EXPECTED_SENTENCE_COUNT = 12
EXPECTED_SENTENCE_CHUNKS = 3
EXPECTED_HEADER_CHUNKS = 6


@pytest.fixture
def sentence_fixture_text() -> str:
    return SENTENCE_FIXTURE.read_text(encoding="utf-8").strip()


@pytest.fixture
def header_aware_fixture_text() -> str:
    return HEADER_AWARE_FIXTURE.read_text(encoding="utf-8").strip()


class TestChunkingFixturesOnDisk:
    def test_sentence_fixture_size_is_in_target_range(self) -> None:
        size = SENTENCE_FIXTURE.stat().st_size
        assert 4000 <= size <= 5200

    def test_header_aware_fixture_size_is_in_target_range(self) -> None:
        size = HEADER_AWARE_FIXTURE.stat().st_size
        assert 7000 <= size <= 9000

    def test_sentence_fixture_has_twelve_numbered_sentences(
        self, sentence_fixture_text: str
    ) -> None:
        sentences = [part for part in sentence_fixture_text.split("\n\n") if part.strip()]
        assert len(sentences) == EXPECTED_SENTENCE_COUNT
        for index, sentence in enumerate(sentences, start=1):
            assert sentence.startswith(f"{index}. ")
            assert sentence.endswith(".")
            assert 300 <= len(sentence) <= 350


class TestSentenceChunkerFixture:
    def test_produces_expected_chunk_count(
        self, sentence_fixture_text: str
    ) -> None:
        chunks = SentenceChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=SENTENCE_FIXTURE.name,
                extracted_text=sentence_fixture_text,
            )
        )

        assert len(chunks) == EXPECTED_SENTENCE_CHUNKS
        assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)

    def test_never_splits_original_sentences(
        self, sentence_fixture_text: str
    ) -> None:
        sentences = sentence_fixture_text.split("\n\n")
        chunks = SentenceChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=SENTENCE_FIXTURE.name,
                extracted_text=sentence_fixture_text,
            )
        )

        for sentence in sentences:
            assert any(sentence in chunk.text for chunk in chunks)


class TestHeaderAwareChunkerFixture:
    def test_produces_expected_chunk_count_and_headers(
        self, header_aware_fixture_text: str
    ) -> None:
        chunks = HeaderAwareChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=HEADER_AWARE_FIXTURE.name,
                extracted_text=header_aware_fixture_text,
            )
        )

        assert len(chunks) == EXPECTED_HEADER_CHUNKS
        assert [chunk.text.split("\n", 1)[0] for chunk in chunks] == [
            "# Introduction",
            "# Embeddings",
            "# Embeddings",
            "# Retrieval",
            "# Retrieval",
            "# Conclusion",
        ]

    def test_introduction_and_conclusion_remain_single_chunks(
        self, header_aware_fixture_text: str
    ) -> None:
        chunks = HeaderAwareChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=HEADER_AWARE_FIXTURE.name,
                extracted_text=header_aware_fixture_text,
            )
        )

        assert chunks[0].text.startswith("# Introduction\n\n")
        assert len(chunks[0].text) <= CHUNK_SIZE
        assert chunks[-1].text.startswith("# Conclusion\n\n")
        assert len(chunks[-1].text) <= CHUNK_SIZE

    def test_split_sections_repeat_header_prefix(
        self, header_aware_fixture_text: str
    ) -> None:
        chunks = HeaderAwareChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=HEADER_AWARE_FIXTURE.name,
                extracted_text=header_aware_fixture_text,
            )
        )

        embeddings = [chunk for chunk in chunks if chunk.text.startswith("# Embeddings")]
        retrieval = [chunk for chunk in chunks if chunk.text.startswith("# Retrieval")]

        assert len(embeddings) == 2
        assert len(retrieval) == 2
        assert all(chunk.text.startswith("# Embeddings\n\n") for chunk in embeddings)
        assert all(chunk.text.startswith("# Retrieval\n\n") for chunk in retrieval)

    def test_split_sections_preserve_full_section_body(
        self, header_aware_fixture_text: str
    ) -> None:
        from app.chunking.header_segmentation import segment_by_headers

        sections = segment_by_headers(header_aware_fixture_text)
        embeddings_body = next(body for header, body in sections if header == "# Embeddings")
        retrieval_body = next(body for header, body in sections if header == "# Retrieval")

        chunks = HeaderAwareChunker().chunk(
            ParsedDocument(
                document_id="fixture",
                filename=HEADER_AWARE_FIXTURE.name,
                extracted_text=header_aware_fixture_text,
            )
        )

        embeddings_indexed = "\n".join(
            chunk.text.removeprefix("# Embeddings\n\n")
            for chunk in chunks
            if chunk.text.startswith("# Embeddings")
        )
        retrieval_indexed = "\n".join(
            chunk.text.removeprefix("# Retrieval\n\n")
            for chunk in chunks
            if chunk.text.startswith("# Retrieval")
        )

        for body, indexed in (
            (embeddings_body, embeddings_indexed),
            (retrieval_body, retrieval_indexed),
        ):
            midpoint = len(body) // 2
            assert body[:200] in indexed
            assert body[midpoint : midpoint + 200] in indexed
            assert body[-200:] in indexed
