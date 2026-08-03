from app.models.rag_config import VectorStoreType
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.chroma_vector_store import ChromaVectorStore


class VectorStoreFactory:
    """Creates vector store implementations."""

    _STORES: dict[VectorStoreType, type[ChromaVectorStore]] = {
        VectorStoreType.CHROMA: ChromaVectorStore,
    }

    @classmethod
    def create(cls, store_type: VectorStoreType) -> BaseVectorStore:
        store_class = cls._STORES.get(store_type)
        if store_class is None:
            raise ValueError(f"Unsupported vector store: {store_type}")
        return store_class()
