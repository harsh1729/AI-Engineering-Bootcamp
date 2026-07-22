from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """One text segment produced from a ParsedDocument by a chunking strategy.

    Embeddings and vector storage consume chunks without needing to know
    which algorithm split the source document.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
