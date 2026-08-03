import logging

from app.context.context_models import ContextPrompt, ContextSource
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalResponse
from app.services.rag_debug import log_rag_debug

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class TestRagDebugLogger:
    def test_logs_structured_rag_debug_block(self, caplog) -> None:
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id=DOCUMENT_ID,
            chunk_index=2,
            text="Refunds are available within 30 days of purchase.",
            source_filename="handbook.pdf",
            score=0.42,
        )
        context_prompt = ContextPrompt(
            system_content="System prompt with context:\n\n[1]\nRefunds are available within 30 days of purchase.",
            user_content="Question:\n\nWhat is the refund policy?",
            sources=[
                ContextSource(
                    source_number=1,
                    document_id=DOCUMENT_ID,
                    filename="handbook.pdf",
                )
            ],
            estimated_tokens=25,
        )

        with caplog.at_level(logging.INFO, logger="app.services.rag_debug"):
            log_rag_debug(
                query="What is the refund policy?",
                embedding_model="text-embedding-3-small",
                retrieval_response=RetrievalResponse(chunks=[chunk]),
                context_prompt=context_prompt,
                llm_answer="Refunds are available within 30 days.",
            )

        output = caplog.text
        assert "==================== RAG DEBUG ====================" in output
        assert "User question:" in output
        assert "What is the refund policy?" in output
        assert "Query embedding model used:" in output
        assert "text-embedding-3-small" in output
        assert "Number of retrieved chunks:" in output
        assert "Rank: 1" in output
        assert "Similarity/Distance score: 0.42" in output
        assert "Document ID:" in output
        assert DOCUMENT_ID in output
        assert "Source filename: handbook.pdf" in output
        assert "Chunk index: 2" in output
        assert "Character count:" in output
        assert "First 300 characters:" in output
        assert "Number of chunks selected by the ContextBuilder:" in output
        assert "Final context length:" in output
        assert "Estimated tokens: 25" in output
        assert "First 1000 characters of the final prompt/context sent to the LLM:" in output
        assert "Final LLM response:" in output
        assert "Refunds are available within 30 days." in output

    def test_logs_empty_retrieval(self, caplog) -> None:
        context_prompt = ContextPrompt(
            system_content="System prompt with no context.",
            user_content="Question:\n\nHello?",
            sources=[],
            estimated_tokens=0,
        )

        with caplog.at_level(logging.INFO, logger="app.services.rag_debug"):
            log_rag_debug(
                query="Hello?",
                embedding_model="text-embedding-3-small",
                retrieval_response=RetrievalResponse(chunks=[]),
                context_prompt=context_prompt,
                llm_answer="I do not have enough context.",
            )

        output = caplog.text
        assert "Retrieved chunks: (none)" in output
        assert "Selected chunks: (none)" in output
