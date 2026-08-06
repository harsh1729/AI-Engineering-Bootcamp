"""Shared helpers for embedding provider implementations."""

from collections.abc import Iterable
from typing import cast

from app.embeddings.embedding_exceptions import EmbeddingProviderError
from app.embeddings.embedding_models import (
    EmbeddingBatchResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingVector,
)


def empty_batch_response() -> EmbeddingBatchResponse:
    """Return the standard empty-batch payload without calling a provider."""
    return EmbeddingBatchResponse(
        embeddings=[],
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )


def build_embedding_response(
    batch: EmbeddingBatchResponse,
) -> EmbeddingResponse:
    """Map the first vector from a batch response to a single-text response."""
    vector = batch.embeddings[0]
    return EmbeddingResponse(
        embedding=vector.embedding,
        model=vector.model,
        usage=batch.usage,
    )


def build_batch_from_indexed_items(
    items: Iterable[tuple[int, list[float]]],
    *,
    expected_count: int,
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> EmbeddingBatchResponse:
    """Reorder indexed provider items into input order."""
    results: list[EmbeddingVector | None] = [None] * expected_count

    for index, embedding in items:
        if index < 0 or index >= expected_count:
            raise EmbeddingProviderError(
                f"Embedding response index {index} is out of range for batch size "
                f"{expected_count}"
            )
        if results[index] is not None:
            raise EmbeddingProviderError(f"Duplicate embedding response index {index}")
        results[index] = EmbeddingVector(embedding=embedding, model=model)

    missing_indices = [index for index, result in enumerate(results) if result is None]
    if missing_indices:
        raise EmbeddingProviderError(
            f"Missing embedding response indices: {missing_indices}"
        )

    embeddings = cast(list[EmbeddingVector], results)
    usage = EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    return EmbeddingBatchResponse(embeddings=embeddings, usage=usage)


def build_batch_from_ordered_vectors(
    embeddings: list[list[float]],
    *,
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> EmbeddingBatchResponse:
    """Map an already ordered embedding list into domain models."""
    vectors = [
        EmbeddingVector(embedding=embedding, model=model) for embedding in embeddings
    ]
    usage = EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    return EmbeddingBatchResponse(embeddings=vectors, usage=usage)
