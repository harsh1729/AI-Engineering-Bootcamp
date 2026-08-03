from functools import lru_cache

from app.context.context_builder import ContextBuilder
from app.models.rag_config import DEFAULT_RAG_OPTIONS
from app.services.chat_service import ChatService
from app.services.rag_pipeline_builder import (
    build_document_ingestion_service,
    build_rag_service,
    build_retrieval_service,
)
from app.services.request_scoped_llm_provider import RequestScopedLLMProvider
from app.services.document_ingestion import DocumentIngestionService
from app.retrieval.retrieval_service import RetrievalService


def get_document_ingestion_service() -> DocumentIngestionService:
    return build_document_ingestion_service(DEFAULT_RAG_OPTIONS)


# TODO: Wire into FastAPI Depends() when multiple retrieval services need DI.
def get_retrieval_service() -> RetrievalService:
    return build_retrieval_service(DEFAULT_RAG_OPTIONS)


@lru_cache
def get_chat_service() -> ChatService:
    """Build the chat orchestration layer and its dependencies."""
    llm_provider = RequestScopedLLMProvider()
    context_builder = ContextBuilder()
    rag_service = build_rag_service(DEFAULT_RAG_OPTIONS, llm_provider, context_builder)
    return ChatService(
        rag_service=rag_service,
        llm_provider=llm_provider,
        context_builder=context_builder,
    )
