import logging

from app.context.context_models import ContextPrompt, ContextSource
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalResponse
from app.services.rag_debug import log_rag_debug

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class TestRagDebugLogger:
    def test_logs_structured_rag_debug_block(self, caplog) -> None:
        chunk_text = "Refunds are available within 30 days of purchase."
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id=DOCUMENT_ID,
            chunk_index=2,
            text=chunk_text,
            source_filename="handbook.pdf",
            score=0.42,
        )
        context_prompt = ContextPrompt(
            system_content="System prompt with context:\n\n[1]\n" + chunk_text,
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
                top_k=5,
                retrieval_response=RetrievalResponse(chunks=[chunk]),
                context_prompt=context_prompt,
            )

        output = caplog.text
        assert "RETRIEVAL DEBUG" in output
        assert "What is the refund policy?" in output
        assert "Top K:" in output
        assert "5" in output
        assert "Retrieved Rank #1" in output
        assert "Chunk Index:" in output
        assert "2" in output
        assert "Similarity Score:" in output
        assert "0.42" in output
        assert chunk_text in output
        assert "First 300 characters:" not in output
        assert "FINAL LLM CONTEXT" in output
        assert context_prompt.system_content in output
        assert context_prompt.user_content in output

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
                top_k=3,
                retrieval_response=RetrievalResponse(chunks=[]),
                context_prompt=context_prompt,
            )

        output = caplog.text
        assert "(no chunks retrieved)" in output
        assert "FINAL LLM CONTEXT" in output
        assert context_prompt.system_content in output
