import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.embeddings.embedding_models import EmbeddingVector
from app.services.document_exceptions import VectorDimensionMismatchError
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    _chunk_id_to_point_id,
)
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

COLLECTION_NAME = "ai-rag-openai-1536"
MODEL = "text-embedding-3-small"
DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


def _record(
    *,
    chunk_id: str | None = None,
    document_id: str = DOCUMENT_ID,
    chunk_index: int = 0,
    text: str = "hello world",
    source_filename: str = "handbook.pdf",
    embedding: list[float] | None = None,
) -> VectorStoreRecord:
    return VectorStoreRecord(
        chunk_id=chunk_id or uuid.uuid4().hex,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_filename=source_filename,
        embedding=EmbeddingVector(
            embedding=embedding or [0.1, 0.2, 0.3],
            model=MODEL,
        ),
    )


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=mock_client,
        collection_name=COLLECTION_NAME,
    )


class TestQdrantVectorStoreInit:
    def test_creates_collection_when_missing(self, mock_client: MagicMock) -> None:
        mock_client.collection_exists.return_value = False

        QdrantVectorStore(
            client=mock_client,
            collection_name=COLLECTION_NAME,
            expected_dimension=3,
        )

        mock_client.create_collection.assert_called_once()
        assert mock_client.create_collection.call_args.kwargs["collection_name"] == COLLECTION_NAME

    def test_requires_collection_name(self, mock_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="collection_name is required"):
            QdrantVectorStore(client=mock_client, collection_name="")


class TestQdrantVectorStoreUpsert:
    def test_upsert_maps_points_with_uuid_ids_and_payload(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        chunk_a = uuid.uuid4().hex
        chunk_b = uuid.uuid4().hex
        records = [
            _record(chunk_id=chunk_a, chunk_index=0, text="first"),
            _record(
                chunk_id=chunk_b,
                chunk_index=1,
                text="second",
                embedding=[0.4, 0.5, 0.6],
            ),
        ]

        store.upsert(records)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME
        points = call_kwargs["points"]
        assert len(points) == 2
        assert str(points[0].id) == _chunk_id_to_point_id(chunk_a)
        assert points[0].payload["text"] == "first"
        assert points[1].vector == [0.4, 0.5, 0.6]

    def test_empty_upsert_is_a_no_op(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        store.upsert([])

        mock_client.upsert.assert_not_called()


class TestQdrantVectorStoreQuery:
    def test_query_maps_points_to_distance_ordered_results(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        chunk_a = uuid.uuid4().hex
        chunk_b = uuid.uuid4().hex
        mock_client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=_chunk_id_to_point_id(chunk_b),
                    score=0.88,
                    payload={
                        "document_id": DOCUMENT_ID,
                        "chunk_index": 1,
                        "source_filename": "handbook.pdf",
                        "text": "second chunk",
                    },
                ),
                SimpleNamespace(
                    id=_chunk_id_to_point_id(chunk_a),
                    score=0.66,
                    payload={
                        "document_id": DOCUMENT_ID,
                        "chunk_index": 0,
                        "source_filename": "handbook.pdf",
                        "text": "first chunk",
                    },
                ),
            ]
        )

        results = store.query([0.9, 0.1, 0.3], limit=2)

        mock_client.query_points.assert_called_once_with(
            collection_name=COLLECTION_NAME,
            query=[0.9, 0.1, 0.3],
            query_filter=None,
            limit=2,
            with_payload=True,
        )
        assert len(results) == 2
        assert results[0].chunk_id == chunk_b
        assert results[0].distance == pytest.approx(0.12)
        assert results[1].chunk_id == chunk_a
        assert results[1].distance == pytest.approx(0.34)

    def test_query_filters_by_document_ids(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.query([0.1, 0.2, 0.3], limit=5, document_ids=["doc-a", "doc-b"])

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert query_filter is not None
        assert query_filter.must[0].key == "document_id"

    def test_query_returns_empty_list_when_no_matches(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        assert store.query([0.1, 0.2], limit=5) == []


class TestQdrantVectorStoreDelete:
    def test_delete_document_filters_by_document_id(
        self,
        store: QdrantVectorStore,
        mock_client: MagicMock,
    ) -> None:
        store.delete_document(DOCUMENT_ID)

        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME
        points_selector = call_kwargs["points_selector"]
        assert points_selector.filter.must[0].key == "document_id"
        assert points_selector.filter.must[0].match.value == DOCUMENT_ID


class TestQdrantVectorStoreDimensionValidation:
    def test_upsert_rejects_wrong_dimension(
        self,
        mock_client: MagicMock,
    ) -> None:
        store = QdrantVectorStore(
            client=mock_client,
            collection_name=COLLECTION_NAME,
            expected_dimension=3,
        )

        with pytest.raises(VectorDimensionMismatchError, match="expected dimension 3"):
            store.upsert([_record(embedding=[0.1, 0.2])])

        mock_client.upsert.assert_not_called()

    def test_query_rejects_wrong_dimension(
        self,
        mock_client: MagicMock,
    ) -> None:
        store = QdrantVectorStore(
            client=mock_client,
            collection_name=COLLECTION_NAME,
            expected_dimension=3,
        )

        with pytest.raises(VectorDimensionMismatchError, match="query embedding"):
            store.query([0.1, 0.2], limit=1)

        mock_client.query_points.assert_not_called()


class TestQdrantVectorStoreDependencyInjection:
    def test_concrete_store_is_substitutable_as_base_vector_store(
        self,
        mock_client: MagicMock,
    ) -> None:
        vector_store: BaseVectorStore = QdrantVectorStore(
            client=mock_client,
            collection_name=COLLECTION_NAME,
        )

        vector_store.upsert([_record()])

        mock_client.upsert.assert_called_once()
