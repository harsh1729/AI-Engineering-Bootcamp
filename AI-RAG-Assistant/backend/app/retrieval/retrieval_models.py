from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Input for a similarity search over indexed document chunks."""

    query: str
    top_k: int = Field(default=5, gt=0)
    # TODO: document_ids: list[str] | None = None — restrict search to specific documents
    # TODO: metadata_filters: dict[str, str] | None = None — filter by chunk metadata
    # TODO: minimum_score: float | None = None — drop results below this score threshold
    # TODO: namespace: str | None = None — scope search to a logical partition
    # TODO: tenant_id: str | None = None — scope search to a tenant


class RetrievedChunk(BaseModel):
    """One chunk returned from retrieval, ranked by similarity."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    score: float


class RetrievalResponse(BaseModel):
    """Ranked chunks retrieved for a query."""

    chunks: list[RetrievedChunk]
