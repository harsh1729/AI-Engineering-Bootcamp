from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class ParsedDocument(BaseModel):
    """Plain-text extraction result for one uploaded document.

    Produced by DocumentParser. Later pipeline stages (chunking, embeddings)
    will consume this without needing to know how the bytes were decoded.
    """

    document_id: str
    filename: str
    extracted_text: str
