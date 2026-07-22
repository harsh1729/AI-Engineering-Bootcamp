from pydantic import BaseModel, computed_field


class EmbeddingRequest(BaseModel):
    """Input for a single text embedding operation.

    Produced by upstream pipeline stages (e.g. chunk text) and consumed by
    an embedding provider without carrying document or chunk identifiers.
    """

    text: str


class EmbeddingUsage(BaseModel):
    """Token accounting returned alongside an embedding result.

    Mirrors the shape of provider usage metadata so services can log cost
    and enforce limits without binding to a specific vendor SDK.
    """

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Vector representation of embedded text plus model metadata.

    Returned by embedding providers and consumed by vector storage and
    retrieval stages without needing to know which API produced the vector.
    """

    embedding: list[float]
    model: str
    usage: EmbeddingUsage

    @computed_field
    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.embedding)
