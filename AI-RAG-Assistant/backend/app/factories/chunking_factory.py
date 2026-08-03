from app.chunking.base_chunker import BaseChunker
from app.chunking.character_chunker import CharacterChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.models.rag_config import ChunkingStrategy


class ChunkingFactory:
    """Creates chunker implementations from a configured strategy."""

    _STRATEGIES: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.CHARACTER: CharacterChunker,
    }

    @classmethod
    def create(cls, strategy: ChunkingStrategy) -> BaseChunker:
        chunker_class = cls._STRATEGIES.get(strategy)
        if chunker_class is None:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        return chunker_class()
