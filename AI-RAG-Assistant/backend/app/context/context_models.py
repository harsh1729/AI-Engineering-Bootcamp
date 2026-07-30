from pydantic import BaseModel

from app.retrieval.retrieval_models import RetrievedChunk


class ContextRequest(BaseModel):
    """Input for building LLM-ready context from retrieved chunks."""

    chunks: list[RetrievedChunk]


class ContextSource(BaseModel):
    """Metadata for one numbered source in the built context."""

    source_number: int
    document_id: str
    filename: str


class ContextResponse(BaseModel):
    """Formatted context and source metadata ready for downstream prompt assembly."""

    context: str
    sources: list[ContextSource]
    estimated_tokens: int
