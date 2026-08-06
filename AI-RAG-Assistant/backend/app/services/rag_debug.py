"""Temporary structured logging for RAG pipeline inspection."""

import logging

from app.context.context_models import ContextPrompt
from app.retrieval.retrieval_models import RetrievalResponse

logger = logging.getLogger(__name__)

_HEADER = "=" * 58


def log_rag_debug(
    *,
    query: str,
    top_k: int,
    retrieval_response: RetrievalResponse,
    context_prompt: ContextPrompt,
) -> None:
    """Emit structured retrieval and LLM context debug blocks. Gated by RAG_DEBUG."""
    retrieved_chunks = retrieval_response.chunks
    final_prompt = (
        f"{context_prompt.system_content}\n\n{context_prompt.user_content}"
    )

    lines = [
        _HEADER,
        "RETRIEVAL DEBUG",
        _HEADER,
        "",
        "Query:",
        query,
        "Top K:",
        str(top_k),
        "",
    ]

    if retrieved_chunks:
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            lines.extend(
                [
                    f"Retrieved Rank #{rank}",
                    "Chunk Index:",
                    str(chunk.chunk_index),
                    "Similarity Score:",
                    str(chunk.score),
                    "Characters:",
                    str(len(chunk.text)),
                    chunk.text,
                    "",
                ]
            )
    else:
        lines.append("(no chunks retrieved)")
        lines.append("")

    lines.extend(
        [
            _HEADER,
            "FINAL LLM CONTEXT",
            _HEADER,
            final_prompt,
            _HEADER,
            "",
        ]
    )

    logger.info("\n".join(lines))
