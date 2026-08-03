from llm_sdk.enums import MessageRole
from llm_sdk.models import LLMMessage, LLMRequest
from llm_sdk.providers.llm_provider import LLMProvider

from app.config import EMBEDDING_MODEL, RAG_DEBUG
from app.context.context_builder import ContextBuilder, unique_document_sources
from app.context.context_models import ContextRequest
from app.models.rag import RAGRequest, RAGResponse
from app.retrieval.retrieval_models import RetrievalRequest
from app.retrieval.retrieval_service import RetrievalService
from app.services.rag_debug import log_rag_debug


def _truncation_warning(finish_reason: str | None) -> str | None:
    reason = (finish_reason or "").lower()
    if "max" in reason and "token" in reason:
        return "The response was cut off because the token limit was reached."
    return None


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

    def ask(self, request: RAGRequest) -> RAGResponse:
        """Retrieve context, generate an answer via the LLM SDK, and return sources."""
        retrieval_response = self._retrieval_service.retrieve(
            self._build_retrieval_request(request)
        )
        context_prompt = self._context_builder.build_prompt(
            ContextRequest(chunks=retrieval_response.chunks),
            request.query,
        )
        llm_response = self._llm_provider.generate_response(
            self._build_llm_request(request, context_prompt.system_content, context_prompt.user_content)
        )

        answer = llm_response.text or ""

        if RAG_DEBUG:
            log_rag_debug(
                query=request.query,
                embedding_model=EMBEDDING_MODEL,
                retrieval_response=retrieval_response,
                context_prompt=context_prompt,
                llm_answer=answer,
            )

        return RAGResponse(
            answer=answer,
            sources=unique_document_sources(context_prompt.sources),
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
