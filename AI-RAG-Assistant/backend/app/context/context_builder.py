import math

from app.context.context_models import (
    ContextPrompt,
    ContextRequest,
    ContextResponse,
    ContextSource,
)
from app.retrieval.retrieval_models import RetrievedChunk

_SOURCE_SEPARATOR = "--------------------------------------------------"

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant.\n\n"
    "Answer ONLY using the supplied context.\n"
    "If the answer is not contained in the context, say you can't find the information in the document.\n"
    "Do not hallucinate."
)


class ContextBuilder:
    """Format retrieved chunks into structured context and LLM prompts."""

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

    def build_prompt(self, request: ContextRequest, query: str) -> ContextPrompt:
        """Build the complete system and user prompt content for grounded generation."""
        context_response = self.build(request)

        if context_response.context:
            system_content = (
                f"{_SYSTEM_INSTRUCTION}\n\nContext:\n\n{context_response.context}"
            )
        else:
            system_content = (
                f"{_SYSTEM_INSTRUCTION}\n\nNo relevant document context was retrieved."
            )

        return ContextPrompt(
            system_content=system_content,
            user_content=f"Question:\n\n{query}",
            sources=context_response.sources,
            estimated_tokens=context_response.estimated_tokens,
        )

    def _format_section(self, source_number: int, chunk: RetrievedChunk) -> str:
        return f"[{source_number}]\n{chunk.text}"

    def _estimate_tokens(self, context: str) -> int:
        # TODO: replace with provider-specific token counting (e.g. tiktoken)
        return math.ceil(len(context) / 4)


def unique_document_sources(sources: list[ContextSource]) -> list[ContextSource]:
    """Return one entry per document for citation UI (no duplicate filenames)."""
    unique: list[ContextSource] = []
    seen_document_ids: set[str] = set()

    for source in sources:
        if source.document_id in seen_document_ids:
            continue
        seen_document_ids.add(source.document_id)
        unique.append(source)

    return [
        ContextSource(
            source_number=index,
            document_id=source.document_id,
            filename=source.filename,
        )
        for index, source in enumerate(unique, start=1)
    ]
