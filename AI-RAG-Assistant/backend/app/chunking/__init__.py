"""Document chunking strategies and models.

Concrete chunkers and a registry will be added in later pipeline phases.
Callers should depend on BaseChunker, not specific implementations.
"""

from app.chunking.base_chunker import BaseChunker
from app.chunking.chunk_models import DocumentChunk

__all__ = [
    "BaseChunker",
    "DocumentChunk",
]
