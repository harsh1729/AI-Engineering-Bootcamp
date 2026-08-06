import logging
from unittest.mock import MagicMock, create_autospec

import pytest
from llm_sdk.enums import MessageRole, ProviderType
from llm_sdk.models import LLMMessage, LLMRequest, LLMResponse
from llm_sdk.providers.llm_provider import LLMProvider

from app.context.context_builder import ContextBuilder
from app.context.context_models import ContextPrompt, ContextRequest, ContextResponse, ContextSource
from app.models.rag import RAGRequest, RAGResponse
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalRequest, RetrievalResponse
from app.retrieval.retrieval_service import RetrievalService
from app.services.rag_service import RAGService

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _retrieved_chunk(*, chunk_id: str, text: str, source_filename: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        chunk_index=0,
        text=text,
        source_filename=source_filename,
        score=0.1,
    )


def _context_prompt(*, context: str, query: str) -> ContextPrompt:
    return ContextPrompt(
        system_content=f"System prompt with context:\n\n{context}",
        user_content=f"Question:\n\n{query}",
        sources=[
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID,
                filename="handbook.pdf",
            )
        ],
        estimated_tokens=10,
    )


@pytest.fixture
def retrieval_service() -> MagicMock:
    return create_autospec(RetrievalService, instance=True)


@pytest.fixture
def context_builder() -> MagicMock:
    return create_autospec(ContextBuilder, instance=True)


@pytest.fixture
def llm_provider() -> MagicMock:
    return create_autospec(LLMProvider, instance=True)


@pytest.fixture
def rag_service(
    retrieval_service: MagicMock,
    context_builder: MagicMock,
    llm_provider: MagicMock,
) -> RAGService:
    return RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        llm_provider=llm_provider,
    )


class TestRAGServiceAsk:
    def test_ask_orchestrates_retrieval_context_and_llm(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
    ) -> None:
        chunk = _retrieved_chunk(
            chunk_id="chunk-1",
            text="Refunds are available within 30 days.",
            source_filename="handbook.pdf",
        )
        context_prompt = _context_prompt(
            context="[1]\nRefunds are available within 30 days.",
            query="What is the refund policy?",
        )
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[chunk])
        context_builder.build_prompt.return_value = context_prompt

        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-1",
            provider="openai",
            model="gpt-4o-mini",
            text="Refunds are available within 30 days.",
            finish_reason="stop",
        )

        response = rag_service.ask(
            RAGRequest(
                query="What is the refund policy?",
                document_ids=[DOCUMENT_ID],
                top_k=3,
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
            )
        )

        retrieval_service.retrieve.assert_called_once_with(
            RetrievalRequest(
                query="What is the refund policy?",
                top_k=3,
                document_ids=[DOCUMENT_ID],
            )
        )
        context_builder.build_prompt.assert_called_once_with(
            ContextRequest(chunks=[chunk]),
            "What is the refund policy?",
        )
        llm_provider.generate_response.assert_called_once_with(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role=MessageRole.SYSTEM,
                        content=context_prompt.system_content,
                    ),
                    LLMMessage(
                        role=MessageRole.USER,
                        content=context_prompt.user_content,
                    ),
                ],
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
            )
        )
        assert response == RAGResponse(
            answer="Refunds are available within 30 days.",
            sources=context_prompt.sources,
            warning=None,
        )

    def test_ask_skips_rag_debug_when_flag_disabled(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ) -> None:
        monkeypatch.setattr("app.services.rag_service.RAG_DEBUG", False)
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[])
        context_builder.build_prompt.return_value = ContextPrompt(
            system_content="System",
            user_content="Question:\n\nHello?",
            sources=[],
            estimated_tokens=0,
        )
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-debug-off",
            provider="openai",
            model="gpt-4o-mini",
            text="Hello",
            finish_reason="stop",
        )

        with caplog.at_level(logging.INFO):
            rag_service.ask(RAGRequest(query="Hello?"))

        assert "RETRIEVAL DEBUG" not in caplog.text

    def test_ask_emits_rag_debug_when_flag_enabled(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ) -> None:
        monkeypatch.setattr("app.services.rag_service.RAG_DEBUG", True)
        chunk = _retrieved_chunk(
            chunk_id="chunk-1",
            text="Refunds are available within 30 days.",
            source_filename="handbook.pdf",
        )
        context_builder.build_prompt.return_value = _context_prompt(
            context="[1]\nRefunds are available within 30 days.",
            query="What is the refund policy?",
        )
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[chunk])
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-debug-on",
            provider="openai",
            model="gpt-4o-mini",
            text="Refunds are available within 30 days.",
            finish_reason="stop",
        )

        with caplog.at_level(logging.INFO):
            rag_service.ask(RAGRequest(query="What is the refund policy?"))

        assert "RETRIEVAL DEBUG" in caplog.text
        assert "What is the refund policy?" in caplog.text

    def test_ask_uses_default_top_k_when_not_specified(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
    ) -> None:
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[])
        context_builder.build_prompt.return_value = ContextPrompt(
            system_content="System",
            user_content="Question:\n\nHello?",
            sources=[],
            estimated_tokens=0,
        )
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-2",
            provider="gemini",
            model="gemini-2.0-flash",
            text="I do not have enough context.",
            finish_reason="stop",
        )

        rag_service.ask(RAGRequest(query="Hello?"))

        retrieval_service.retrieve.assert_called_once_with(
            RetrievalRequest(query="Hello?")
        )

    def test_ask_returns_truncation_warning(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
    ) -> None:
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[])
        context_builder.build_prompt.return_value = ContextPrompt(
            system_content="System",
            user_content="Question:\n\nSummarize everything",
            sources=[],
            estimated_tokens=0,
        )
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-4",
            provider="claude",
            model="claude-sonnet-4",
            text="Partial answer",
            finish_reason="max_tokens",
        )

        response = rag_service.ask(RAGRequest(query="Summarize everything"))

        assert response.warning == (
            "The response was cut off because the token limit was reached."
        )

    def test_ask_returns_sources_from_context_builder(
        self,
        rag_service: RAGService,
        retrieval_service: MagicMock,
        context_builder: MagicMock,
        llm_provider: MagicMock,
    ) -> None:
        chunk = _retrieved_chunk(
            chunk_id="chunk-1",
            text="Pricing starts at $99.",
            source_filename="pricing.pdf",
        )
        sources = [
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID,
                filename="pricing.pdf",
            )
        ]
        retrieval_service.retrieve.return_value = RetrievalResponse(chunks=[chunk])
        context_builder.build_prompt.return_value = ContextPrompt(
            system_content="System",
            user_content="Question:\n\nHow much does it cost?",
            sources=sources,
            estimated_tokens=12,
        )
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-5",
            provider="openai",
            model="gpt-4o-mini",
            text="Pricing starts at $99.",
            finish_reason="stop",
        )

        response = rag_service.ask(RAGRequest(query="How much does it cost?"))

        assert response.sources == sources
