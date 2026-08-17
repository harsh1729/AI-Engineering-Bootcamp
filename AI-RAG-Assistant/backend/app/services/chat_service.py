from dataclasses import dataclass

from llm_sdk.enums import MessageRole
from llm_sdk.models import LLMRequest
from llm_sdk.providers.llm_provider import LLMProvider

from app.context.context_builder import ContextBuilder
from app.context.context_models import ContextSource
from app.llm.llm_request_builder import build_chat_llm_request
from app.models.chat import ChatRequest, ChatResponse
from app.models.rag import RAGRequest
from app.models.rag_config import RagOptions
from app.services.rag_pipeline_builder import build_rag_service
from app.services.rag_service import RAGService


def _truncation_warning(finish_reason: str | None) -> str | None:
    reason = (finish_reason or "").lower()
    if "max" in reason and "token" in reason:
        return "The response was cut off because the token limit was reached."
    return None


@dataclass(frozen=True)
class PreparedStreamRequest:
    llm_request: LLMRequest
    sources: list[ContextSource]
    rag_service: RAGService | None = None


class ChatService:
    """Routes chat requests to direct LLM generation or RAG-backed generation."""

    def __init__(
        self,
        rag_service: RAGService,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._default_rag_service = rag_service
        self._llm_provider = llm_provider
        self._context_builder = context_builder or ContextBuilder()

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request, using RAG when document IDs are attached."""
        if request.document_ids:
            return self._rag_chat(request)
        return self._direct_chat(request)

    def prepare_stream(self, request: ChatRequest) -> PreparedStreamRequest:
        """Build the LLM request (and optional RAG service) for streaming."""
        if request.document_ids:
            rag_service = self._resolve_rag_service(request)
            prepared = rag_service.prepare(self._to_rag_request(request))
            return PreparedStreamRequest(
                llm_request=prepared.llm_request,
                sources=prepared.sources,
                rag_service=rag_service,
            )

        return PreparedStreamRequest(
            llm_request=build_chat_llm_request(request, include_tools=True),
            sources=[],
        )

    def _direct_chat(self, request: ChatRequest) -> ChatResponse:
        llm_response = self._llm_provider.generate_response(self._build_llm_request(request))
        return ChatResponse(
            response=llm_response.text or "",
            warning=_truncation_warning(llm_response.finish_reason),
        )

    def _rag_chat(self, request: ChatRequest) -> ChatResponse:
        rag_response = self._resolve_rag_service(request).ask(self._to_rag_request(request))
        return ChatResponse(
            response=rag_response.answer,
            warning=rag_response.warning,
            sources=rag_response.sources,
        )

    def _resolve_rag_service(self, request: ChatRequest) -> RAGService:
        if request.document_ids:
            if request.rag_options is None:
                raise ValueError(
                    "RAG options must be resolved from attached documents before chat."
                )
            return build_rag_service(
                request.rag_options,
                self._llm_provider,
                self._context_builder,
            )

        rag_options = request.rag_options
        if rag_options is None:
            return self._default_rag_service
        return build_rag_service(rag_options, self._llm_provider, self._context_builder)

    def _build_llm_request(self, request: ChatRequest):
        return build_chat_llm_request(request, include_tools=True)

    def _to_rag_request(self, request: ChatRequest) -> RAGRequest:
        return RAGRequest(
            query=self._extract_latest_user_query(request),
            document_ids=request.document_ids,
            provider=request.provider,
            model=request.model,
        )

    def _extract_latest_user_query(self, request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == MessageRole.USER:
                return message.content
        raise ValueError("At least one user message is required.")
