from app.chunking.base_chunker import BaseChunker
from app.chunking.character_chunker import CharacterChunker
from app.chunking.header_aware_chunker import HeaderAwareChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.models.rag_config import ChunkingStrategy


class ChunkingFactory:
    """Creates chunker implementations from a configured strategy."""

    _STRATEGIES: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.CHARACTER: CharacterChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.HEADER_AWARE: HeaderAwareChunker,
    }

    @classmethod
    def create(cls, strategy: ChunkingStrategy) -> BaseChunker:
        chunker_class = cls._STRATEGIES.get(strategy)
        if chunker_class is None:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        return chunker_class()
