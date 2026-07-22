from abc import ABC, abstractmethod

from app.chunking.chunk_models import DocumentChunk
from app.models.document import ParsedDocument


class BaseChunker(ABC):
    """Strategy interface for splitting a ParsedDocument into chunks."""

    @abstractmethod
    def chunk(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        """Return ordered chunks derived from `parsed_document`.

        Each implementation chooses its own splitting rules (character
        windows, token limits, markdown structure, semantic boundaries, etc.).
        """
