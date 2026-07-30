from unittest.mock import MagicMock, create_autospec

import pytest

from app.embeddings.embedding_models import EmbeddingResponse, EmbeddingUsage
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.base_retriever import BaseRetriever
from app.retrieval.retrieval_models import RetrievedChunk, RetrievalRequest, RetrievalResponse
from app.retrieval.retrieval_service import RetrievalService
from app.retrieval.similarity_retriever import SimilarityRetriever
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult

MODEL = "text-embedding-3-small"


def _embedding_response(*, embedding: list[float] | None = None) -> EmbeddingResponse:
    return EmbeddingResponse(
        embedding=embedding or [0.1, 0.2, 0.3],
        model=MODEL,
        usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3),
    )


def _vector_results() -> list[VectorQueryResult]:
    return [
        VectorQueryResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="first chunk",
            source_filename="handbook.pdf",
            distance=0.12,
        ),
        VectorQueryResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            text="second chunk",
            source_filename="handbook.pdf",
            distance=0.34,
        ),
    ]


@pytest.fixture
def embedding_service() -> MagicMock:
    return create_autospec(EmbeddingService, instance=True)


@pytest.fixture
def vector_store() -> MagicMock:
    return create_autospec(BaseVectorStore, instance=True)


@pytest.fixture
def similarity_retriever(
    embedding_service: MagicMock,
    vector_store: MagicMock,
) -> SimilarityRetriever:
    return SimilarityRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


class TestRetrievalRequest:
    def test_top_k_defaults_to_five(self) -> None:
        request = RetrievalRequest(query="hello")

        assert request.top_k == 5


class TestSimilarityRetriever:
    def test_retrieve_embeds_query_and_searches_vector_store(
        self,
        similarity_retriever: SimilarityRetriever,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        embedding_service.embed.return_value = _embedding_response()
        vector_store.query.return_value = _vector_results()
        request = RetrievalRequest(query="What is RAG?", top_k=2)

        response = similarity_retriever.retrieve(request)

        embedding_service.embed.assert_called_once_with("What is RAG?")
        vector_store.query.assert_called_once_with([0.1, 0.2, 0.3], 2)
        assert response == RetrievalResponse(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    chunk_index=0,
                    text="first chunk",
                    source_filename="handbook.pdf",
                    score=0.12,
                ),
                RetrievedChunk(
                    chunk_id="chunk-2",
                    document_id="doc-1",
                    chunk_index=1,
                    text="second chunk",
                    source_filename="handbook.pdf",
                    score=0.34,
                ),
            ]
        )

    def test_retrieve_preserves_vector_store_order(
        self,
        similarity_retriever: SimilarityRetriever,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        embedding_service.embed.return_value = _embedding_response()
        vector_store.query.return_value = _vector_results()

        response = similarity_retriever.retrieve(
            RetrievalRequest(query="search text", top_k=5)
        )

        assert [chunk.chunk_id for chunk in response.chunks] == ["chunk-1", "chunk-2"]
        assert [chunk.score for chunk in response.chunks] == [0.12, 0.34]

    def test_retrieve_uses_default_top_k_when_not_specified(
        self,
        similarity_retriever: SimilarityRetriever,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        embedding_service.embed.return_value = _embedding_response()
        vector_store.query.return_value = []

        similarity_retriever.retrieve(RetrievalRequest(query="search text"))

        vector_store.query.assert_called_once_with([0.1, 0.2, 0.3], 5)

    def test_retrieve_returns_empty_response_when_vector_store_has_no_matches(
        self,
        similarity_retriever: SimilarityRetriever,
        embedding_service: MagicMock,
        vector_store: MagicMock,
    ) -> None:
        embedding_service.embed.return_value = _embedding_response()
        vector_store.query.return_value = []

        response = similarity_retriever.retrieve(
            RetrievalRequest(query="missing topic", top_k=3)
        )

        assert response == RetrievalResponse(chunks=[])


class TestRetrievalService:
    def test_retrieve_delegates_to_injected_retriever(self) -> None:
        retriever = create_autospec(BaseRetriever, instance=True)
        expected = RetrievalResponse(chunks=[])
        retriever.retrieve.return_value = expected
        service = RetrievalService(retriever)
        request = RetrievalRequest(query="hello", top_k=1)

        response = service.retrieve(request)

        retriever.retrieve.assert_called_once_with(request)
        assert response is expected

    def test_service_accepts_any_base_retriever_implementation(
        self,
        similarity_retriever: SimilarityRetriever,
    ) -> None:
        service: RetrievalService = RetrievalService(similarity_retriever)

        assert isinstance(service, RetrievalService)
