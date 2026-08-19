from app.config import (
    resolve_pgvector_dimension,
    resolve_pgvector_table,
    resolve_pinecone_dimension,
    resolve_pinecone_index,
    resolve_qdrant_collection,
    resolve_qdrant_dimension,
)
from app.models.rag_config import EmbeddingProviderType, VectorStoreType
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.chroma_vector_store import ChromaVectorStore
from app.vector_store.pgvector_vector_store import PgVectorVectorStore
from app.vector_store.pinecone_vector_store import PineconeVectorStore
from app.vector_store.qdrant_vector_store import QdrantVectorStore


class VectorStoreFactory:
    """Creates vector store implementations."""

    @classmethod
    def create(
        cls,
        store_type: VectorStoreType,
        *,
        embedding_provider: EmbeddingProviderType | None = None,
    ) -> BaseVectorStore:
        if store_type == VectorStoreType.PINECONE:
            if embedding_provider is None:
                raise ValueError("embedding_provider is required for Pinecone vector store.")
            provider_value = embedding_provider.value
            return PineconeVectorStore(
                index_name=resolve_pinecone_index(provider_value),
                expected_dimension=resolve_pinecone_dimension(provider_value),
            )

        if store_type == VectorStoreType.PGVECTOR:
            if embedding_provider is None:
                raise ValueError("embedding_provider is required for pgvector vector store.")
            provider_value = embedding_provider.value
            return PgVectorVectorStore(
                table_name=resolve_pgvector_table(provider_value),
                expected_dimension=resolve_pgvector_dimension(provider_value),
            )

        if store_type == VectorStoreType.QDRANT:
            if embedding_provider is None:
                raise ValueError("embedding_provider is required for Qdrant vector store.")
            provider_value = embedding_provider.value
            return QdrantVectorStore(
                collection_name=resolve_qdrant_collection(provider_value),
                expected_dimension=resolve_qdrant_dimension(provider_value),
            )

        if store_type == VectorStoreType.CHROMA:
            return ChromaVectorStore()

        raise ValueError(f"Unsupported vector store: {store_type}")
