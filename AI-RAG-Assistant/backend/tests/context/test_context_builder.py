import math

import pytest

from app.context.context_builder import ContextBuilder, _SOURCE_SEPARATOR
from app.context.context_models import ContextRequest, ContextResponse, ContextSource
from app.retrieval.retrieval_models import RetrievedChunk

DOCUMENT_ID_A = "11111111-2222-3333-4444-555555555555"
DOCUMENT_ID_B = "22222222-3333-4444-5555-666666666666"


def _chunk(
    *,
    chunk_id: str,
    document_id: str = DOCUMENT_ID_A,
    chunk_index: int = 0,
    text: str,
    source_filename: str,
    score: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_filename=source_filename,
        score=score,
    )


@pytest.fixture
def builder() -> ContextBuilder:
    return ContextBuilder()


class TestContextBuilderEmpty:
    def test_empty_chunk_list_returns_empty_context(
        self,
        builder: ContextBuilder,
    ) -> None:
        response = builder.build(ContextRequest(chunks=[]))

        assert response == ContextResponse(context="", sources=[], estimated_tokens=0)


class TestContextBuilderSingleChunk:
    def test_single_chunk_formatting(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunk = _chunk(
            chunk_id="chunk-1",
            text="Refunds are available within 30 days.",
            source_filename="refund_policy.pdf",
        )

        response = builder.build(ContextRequest(chunks=[chunk]))

        assert response.context == "[1]\nRefunds are available within 30 days."
        assert response.sources == [
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID_A,
                filename="refund_policy.pdf",
            )
        ]

    def test_single_chunk_estimated_tokens_is_positive(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunk = _chunk(
            chunk_id="chunk-1",
            text="Some context text.",
            source_filename="notes.txt",
        )

        response = builder.build(ContextRequest(chunks=[chunk]))

        assert response.estimated_tokens > 0
        assert response.estimated_tokens == math.ceil(len(response.context) / 4)


class TestContextBuilderMultipleChunks:
    def test_multiple_chunks_use_separator_and_numbering(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunks = [
            _chunk(
                chunk_id="chunk-1",
                text="Refunds are available within 30 days.",
                source_filename="refund_policy.pdf",
            ),
            _chunk(
                chunk_id="chunk-2",
                document_id=DOCUMENT_ID_B,
                chunk_index=1,
                text="Annual plans start at $99.",
                source_filename="pricing.pdf",
            ),
        ]

        response = builder.build(ContextRequest(chunks=chunks))

        assert response.context == (
            "[1]\nRefunds are available within 30 days."
            f"\n\n{_SOURCE_SEPARATOR}\n\n"
            "[2]\nAnnual plans start at $99."
        )
        assert response.sources == [
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID_A,
                filename="refund_policy.pdf",
            ),
            ContextSource(
                source_number=2,
                document_id=DOCUMENT_ID_B,
                filename="pricing.pdf",
            ),
        ]

    def test_retrieval_order_is_preserved(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunks = [
            _chunk(
                chunk_id="chunk-a",
                chunk_index=0,
                text="First retrieved.",
                source_filename="a.pdf",
                score=0.05,
            ),
            _chunk(
                chunk_id="chunk-b",
                chunk_index=1,
                text="Second retrieved.",
                source_filename="b.pdf",
                score=0.15,
            ),
            _chunk(
                chunk_id="chunk-c",
                chunk_index=2,
                text="Third retrieved.",
                source_filename="c.pdf",
                score=0.25,
            ),
        ]

        response = builder.build(ContextRequest(chunks=chunks))

        assert [source.source_number for source in response.sources] == [1, 2, 3]
        assert [source.filename for source in response.sources] == [
            "a.pdf",
            "b.pdf",
            "c.pdf",
        ]
        assert response.context.index("First retrieved.") < response.context.index(
            "Second retrieved."
        )
        assert response.context.index("Second retrieved.") < response.context.index(
            "Third retrieved."
        )

    def test_estimated_tokens_matches_context_length(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunks = [
            _chunk(
                chunk_id="chunk-1",
                text="Alpha content.",
                source_filename="alpha.pdf",
            ),
            _chunk(
                chunk_id="chunk-2",
                text="Beta content.",
                source_filename="beta.pdf",
            ),
        ]

        response = builder.build(ContextRequest(chunks=chunks))

        assert response.estimated_tokens == math.ceil(len(response.context) / 4)


class TestContextBuilderPrompt:
    def test_build_prompt_includes_system_instruction_and_question(
        self,
        builder: ContextBuilder,
    ) -> None:
        chunk = _chunk(
            chunk_id="chunk-1",
            text="Refunds are available within 30 days.",
            source_filename="refund_policy.pdf",
        )

        prompt = builder.build_prompt(
            ContextRequest(chunks=[chunk]),
            "What is the refund policy?",
        )

        assert "Answer ONLY using the supplied context." in prompt.system_content
        assert "Do not hallucinate." in prompt.system_content
        assert "[1]\nRefunds are available within 30 days." in prompt.system_content
        assert prompt.user_content == "Question:\n\nWhat is the refund policy?"
        assert prompt.sources[0].filename == "refund_policy.pdf"

    def test_build_prompt_handles_empty_retrieval(
        self,
        builder: ContextBuilder,
    ) -> None:
        prompt = builder.build_prompt(ContextRequest(chunks=[]), "Any policies?")

        assert "No relevant document context was retrieved." in prompt.system_content
        assert prompt.sources == []


class TestUniqueDocumentSources:
    def test_deduplicates_sources_by_document_id(self) -> None:
        from app.context.context_builder import unique_document_sources

        sources = unique_document_sources(
            [
                ContextSource(
                    source_number=1,
                    document_id=DOCUMENT_ID_A,
                    filename="roadmap.rtf",
                ),
                ContextSource(
                    source_number=2,
                    document_id=DOCUMENT_ID_A,
                    filename="roadmap.rtf",
                ),
                ContextSource(
                    source_number=3,
                    document_id=DOCUMENT_ID_B,
                    filename="pricing.pdf",
                ),
            ]
        )

        assert sources == [
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID_A,
                filename="roadmap.rtf",
            ),
            ContextSource(
                source_number=2,
                document_id=DOCUMENT_ID_B,
                filename="pricing.pdf",
            ),
        ]
