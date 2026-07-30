import math

from app.context.context_models import ContextRequest, ContextResponse, ContextSource
from app.retrieval.retrieval_models import RetrievedChunk

_SOURCE_SEPARATOR = "--------------------------------------------------"


class ContextBuilder:
    """Format retrieved chunks into structured context for LLM consumption."""

    def build(self, request: ContextRequest) -> ContextResponse:
        """Transform retrieved chunks into formatted context and source metadata."""
        # TODO: token budget trimming — drop or shorten chunks to fit a max token budget
        # TODO: duplicate chunk removal — deduplicate identical or near-identical chunks
        # TODO: chunk merging — combine adjacent chunks from the same document
        # TODO: metadata formatting — include chunk_index, score, or custom metadata in output
        # TODO: citation formatting — emit inline citation markers for downstream prompts
        # TODO: reranked chunks — accept externally reranked order before formatting

        if not request.chunks:
            return ContextResponse(context="", sources=[], estimated_tokens=0)

        sections: list[str] = []
        sources: list[ContextSource] = []

        for source_number, chunk in enumerate(request.chunks, start=1):
            sections.append(self._format_section(source_number, chunk))
            sources.append(
                ContextSource(
                    source_number=source_number,
                    document_id=chunk.document_id,
                    filename=chunk.source_filename,
                )
            )

        context = f"\n\n{_SOURCE_SEPARATOR}\n\n".join(sections)
        estimated_tokens = self._estimate_tokens(context)

        return ContextResponse(
            context=context,
            sources=sources,
            estimated_tokens=estimated_tokens,
        )

    def _format_section(self, source_number: int, chunk: RetrievedChunk) -> str:
        return (
            f"Source [{source_number}]\n"
            f"Filename: {chunk.source_filename}\n\n"
            f"{chunk.text}"
        )

    def _estimate_tokens(self, context: str) -> int:
        # TODO: replace with provider-specific token counting (e.g. tiktoken)
        return math.ceil(len(context) / 4)
