from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    indexed_chunk_count: int


class DocumentSummary(BaseModel):
    id: str
    filename: str
    content_type: str
    chunking_strategy: str
    embedding_provider: str
    vector_db: str
    chunk_count: int
    user_id: str | None = None
    guest_id: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class ParsedDocument(BaseModel):
    """Plain-text extraction result for one uploaded document.

    Produced by DocumentParser. Later pipeline stages (chunking, embeddings)
    will consume this without needing to know how the bytes were decoded.
    """

    document_id: str
    filename: str
    extracted_text: str


class DocumentMetadata(BaseModel):
    """Persistent upload metadata for one stored document."""

    document_id: str
    original_filename: str
    stored_filename: str
