from pydantic import BaseModel, computed_field


class EmbeddingUsage(BaseModel):
    """Token accounting for an embedding request.

    For batch calls this reflects the entire request, not an individual vector.
    """

    prompt_tokens: int
    total_tokens: int


class EmbeddingVector(BaseModel):
    """One embedding vector plus model metadata.

    Per-vector results intentionally omit usage; batch-level usage lives on
    EmbeddingBatchResponse.
    """

    embedding: list[float]
    model: str

    @computed_field
    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.embedding)


class EmbeddingBatchResponse(BaseModel):
    """Result of embedding multiple texts in one provider request.

    `usage` is request-level token accounting shared across all vectors in
    `embeddings`.
    """

    embeddings: list[EmbeddingVector]
    usage: EmbeddingUsage


class EmbeddingResponse(BaseModel):
    """Result of embedding a single text input.

    `usage` reflects the one provider request made for this call.
    """

    embedding: list[float]
    model: str
    usage: EmbeddingUsage

    @computed_field
    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.embedding)
