from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.base_retriever import BaseRetriever
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalRequest, RetrievalResponse
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult


class SimilarityRetriever(BaseRetriever):
    """Retrieve chunks by embedding the query and searching the vector store."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        query_embedding = self._embedding_service.embed(request.query)
        results = self._vector_store.query(query_embedding.embedding, request.top_k)
        return RetrievalResponse(chunks=[self._map_result(result) for result in results])

    def _map_result(self, result: VectorQueryResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            chunk_index=result.chunk_index,
            text=result.text,
            source_filename=result.source_filename,
            score=self._to_score(result.distance),
        )

    def _to_score(self, distance: float) -> float:
        """Convert vector-store distance to a retrieval-layer score.

        Chroma returns cosine distance (lower is more similar). Score conversion
        policy is owned here so store-specific terminology does not leak upward.
        """
        return distance
