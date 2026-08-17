from functools import lru_cache

from app.chunking.chunking_service import ChunkingService
from app.embeddings.embedding_service import EmbeddingService
from app.factories import ChunkingFactory, EmbeddingFactory, VectorStoreFactory
from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)
from app.context.context_builder import ContextBuilder
from app.retrieval.retrieval_service import RetrievalService
from app.retrieval.similarity_retriever import SimilarityRetriever
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_parser import document_parser
from app.services.rag_service import RAGService
from app.vector_store.base_vector_store import BaseVectorStore
from llm_sdk.providers.llm_provider import LLMProvider


@lru_cache
def _get_shared_vector_store(
    store_type: VectorStoreType,
    embedding_provider: EmbeddingProviderType,
) -> BaseVectorStore:
    return VectorStoreFactory.create(
        store_type,
        embedding_provider=embedding_provider,
    )


@lru_cache
def _get_shared_embedding_service(provider: EmbeddingProviderType) -> EmbeddingService:
    return EmbeddingService(EmbeddingFactory.create(provider))


@lru_cache
def _get_shared_chunking_service(strategy: ChunkingStrategy) -> ChunkingService:
    return ChunkingService(ChunkingFactory.create(strategy))


def build_document_ingestion_service(rag_options: RagOptions) -> DocumentIngestionService:
    """Wire ingestion dependencies from the selected RAG options."""
    return DocumentIngestionService(
        document_parser=document_parser,
        chunking_service=_get_shared_chunking_service(rag_options.chunking_strategy),
        embedding_service=_get_shared_embedding_service(rag_options.embedding_provider),
        vector_store=_get_shared_vector_store(
            rag_options.vector_store,
            rag_options.embedding_provider,
        ),
    )


def build_retrieval_service(rag_options: RagOptions) -> RetrievalService:
    """Wire retrieval dependencies from the selected RAG options."""
    return RetrievalService(
        SimilarityRetriever(
            embedding_service=_get_shared_embedding_service(rag_options.embedding_provider),
            vector_store=_get_shared_vector_store(
                rag_options.vector_store,
                rag_options.embedding_provider,
            ),
        )
    )


def build_rag_service(
    rag_options: RagOptions,
    llm_provider: LLMProvider,
    context_builder: ContextBuilder | None = None,
) -> RAGService:
    """Wire the full RAG pipeline from the selected RAG options."""
    return RAGService(
        retrieval_service=build_retrieval_service(rag_options),
        context_builder=context_builder or ContextBuilder(),
        llm_provider=llm_provider,
    )
