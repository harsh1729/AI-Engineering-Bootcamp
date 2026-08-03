"""Temporary structured logging for RAG pipeline inspection."""

import logging

from app.context.context_models import ContextPrompt
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalResponse

logger = logging.getLogger(__name__)

_SEPARATOR = "==================== RAG DEBUG ===================="


def _preview(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def log_rag_debug(
    *,
    query: str,
    embedding_model: str,
    retrieval_response: RetrievalResponse,
    context_prompt: ContextPrompt,
    llm_answer: str,
) -> None:
    """Emit a structured RAG debug block. No-op unless RAG_DEBUG is enabled."""
    retrieved_chunks = retrieval_response.chunks
    final_prompt = (
        f"{context_prompt.system_content}\n\n{context_prompt.user_content}"
    )
    final_context_chars = len(final_prompt)
    final_context_tokens = context_prompt.estimated_tokens

    lines = [
        _SEPARATOR,
        "User question:",
        query,
        "",
        "Query embedding model used:",
        embedding_model,
        "",
        "Number of retrieved chunks:",
        str(len(retrieved_chunks)),
    ]

    if retrieved_chunks:
        lines.append("")
        lines.append("Retrieved chunks:")
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            lines.extend(_format_retrieved_chunk(rank, chunk))
    else:
        lines.append("")
        lines.append("Retrieved chunks: (none)")

    lines.extend(
        [
            "",
            "Number of chunks selected by the ContextBuilder:",
            str(len(context_prompt.sources)),
        ]
    )

    if context_prompt.sources:
        lines.append("")
        lines.append("Selected chunks:")
        for source, chunk in zip(context_prompt.sources, retrieved_chunks):
            lines.extend(
                [
                    f"  Document ID: {source.document_id}",
                    f"  Chunk index: {chunk.chunk_index}",
                    f"  Source filename: {source.filename}",
                    "",
                ]
            )
    else:
        lines.append("")
        lines.append("Selected chunks: (none)")

    lines.extend(
        [
            "Final context length:",
            f"  Characters: {final_context_chars}",
            f"  Estimated tokens: {final_context_tokens}",
            "",
            "First 1000 characters of the final prompt/context sent to the LLM:",
            _preview(final_prompt, 1000),
            "",
            "Final LLM response:",
            llm_answer,
            _SEPARATOR,
        ]
    )

    logger.info("\n".join(lines))


def _format_retrieved_chunk(rank: int, chunk: RetrievedChunk) -> list[str]:
    return [
        f"  Rank: {rank}",
        f"  Similarity/Distance score: {chunk.score}",
        f"  Document ID: {chunk.document_id}",
        f"  Source filename: {chunk.source_filename}",
        f"  Chunk index: {chunk.chunk_index}",
        f"  Character count: {len(chunk.text)}",
        f"  First 300 characters: {_preview(chunk.text, 300)}",
        "",
    ]
