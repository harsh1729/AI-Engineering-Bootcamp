from collections.abc import Generator
from dataclasses import dataclass

from llm_sdk.enums import MessageRole
from llm_sdk.models import LLMMessage, LLMRequest
from llm_sdk.providers.llm_provider import LLMProvider

from app.config import RAG_DEBUG
from app.context.context_builder import ContextBuilder, unique_document_sources
from app.context.context_models import ContextRequest, ContextSource
from app.models.rag import RAGRequest, RAGResponse
from app.retrieval.retrieval_models import RetrievalRequest, RetrievalResponse
from app.retrieval.retrieval_service import RetrievalService
from app.services.rag_debug import log_rag_debug


def _truncation_warning(finish_reason: str | None) -> str | None:
    reason = (finish_reason or "").lower()
    if "max" in reason and "token" in reason:
        return "The response was cut off because the token limit was reached."
    return None


@dataclass(frozen=True)
class PreparedRagContext:
    """Retrieval and prompt assembly finished; ready for LLM generation."""

    llm_request: LLMRequest
    sources: list[ContextSource]
    retrieval_request: RetrievalRequest
    retrieval_response: RetrievalResponse


class RAGService:
    """Orchestrates retrieval, context formatting, prompt assembly, and LLM generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._llm_provider = llm_provider

    def prepare(self, request: RAGRequest) -> PreparedRagContext:
        """Retrieve chunks and assemble the grounded LLM request."""
        retrieval_request = self._build_retrieval_request(request)
        retrieval_response = self._retrieval_service.retrieve(retrieval_request)
        context_prompt = self._context_builder.build_prompt(
            ContextRequest(chunks=retrieval_response.chunks),
            request.query,
        )

        if RAG_DEBUG:
            log_rag_debug(
                query=request.query,
                top_k=retrieval_request.top_k,
                retrieval_response=retrieval_response,
                context_prompt=context_prompt,
            )

        return PreparedRagContext(
            llm_request=self._build_llm_request(
                request,
                context_prompt.system_content,
                context_prompt.user_content,
            ),
            sources=unique_document_sources(context_prompt.sources),
            retrieval_request=retrieval_request,
            retrieval_response=retrieval_response,
        )

    def stream_answer(self, llm_request: LLMRequest) -> Generator[str, None, None]:
        """Stream tokens for a prepared grounded LLM request."""
        for chunk in self._llm_provider.generate_stream(llm_request):
            if chunk.text:
                yield chunk.text

    def ask(self, request: RAGRequest) -> RAGResponse:
        """Retrieve context, generate an answer via the LLM SDK, and return sources."""
        prepared = self.prepare(request)
        llm_response = self._llm_provider.generate_response(prepared.llm_request)

        return RAGResponse(
            answer=llm_response.text or "",
            sources=prepared.sources,
            warning=_truncation_warning(llm_response.finish_reason),
        )

    def _build_retrieval_request(self, request: RAGRequest) -> RetrievalRequest:
        retrieval_kwargs: dict = {"query": request.query}
        if request.top_k is not None:
            retrieval_kwargs["top_k"] = request.top_k
        if request.document_ids:
            retrieval_kwargs["document_ids"] = request.document_ids
        return RetrievalRequest(**retrieval_kwargs)

    def _build_llm_request(
        self,
        request: RAGRequest,
        system_content: str,
        user_content: str,
    ) -> LLMRequest:
        return LLMRequest(
            messages=[
                LLMMessage(role=MessageRole.SYSTEM, content=system_content),
                LLMMessage(role=MessageRole.USER, content=user_content),
            ],
            provider=request.provider,
            model=request.model,
        )
