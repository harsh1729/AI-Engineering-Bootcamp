from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.document_exceptions import PineconeDimensionMismatchError
from app.embeddings.embedding_models import EmbeddingVector
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.pinecone_vector_store import PineconeVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

INDEX_NAME = "test-index"
NAMESPACE = "document-chunks"
MODEL = "text-embedding-3-small"


def _record(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    chunk_index: int = 0,
    text: str = "hello world",
    source_filename: str = "handbook.pdf",
    embedding: list[float] | None = None,
) -> VectorStoreRecord:
    return VectorStoreRecord(
        chunk_id=chunk_id,
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
def mock_index() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_client(mock_index: MagicMock) -> MagicMock:
    client = MagicMock()
    client.Index.return_value = mock_index
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> PineconeVectorStore:
    return PineconeVectorStore(
        client=mock_client,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )


class TestPineconeVectorStoreInit:
    def test_index_is_created_with_configured_name(
        self,
        mock_client: MagicMock,
        mock_index: MagicMock,
    ) -> None:
        PineconeVectorStore(
            client=mock_client,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )

        mock_client.Index.assert_called_once_with(INDEX_NAME)


class TestPineconeVectorStoreUpsert:
    def test_upsert_maps_ids_values_and_metadata(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        records = [
            _record(chunk_id="chunk-a", document_id="doc-a", chunk_index=0, text="first"),
            _record(
                chunk_id="chunk-b",
                document_id="doc-a",
                chunk_index=1,
                text="second",
                embedding=[0.4, 0.5, 0.6],
            ),
        ]

        store.upsert(records)

        mock_index.upsert.assert_called_once_with(
            vectors=[
                {
                    "id": "chunk-a",
                    "values": [0.1, 0.2, 0.3],
                    "metadata": {
                        "document_id": "doc-a",
                        "chunk_index": 0,
                        "source_filename": "handbook.pdf",
                        "text": "first",
                    },
                },
                {
                    "id": "chunk-b",
                    "values": [0.4, 0.5, 0.6],
                    "metadata": {
                        "document_id": "doc-a",
                        "chunk_index": 1,
                        "source_filename": "handbook.pdf",
                        "text": "second",
                    },
                },
            ],
            namespace=NAMESPACE,
        )

    def test_empty_upsert_is_a_no_op(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        store.upsert([])

        mock_index.upsert.assert_not_called()


class TestPineconeVectorStoreQuery:
    def test_query_maps_matches_to_distance_ordered_results(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        mock_index.query.return_value = SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id="chunk-2",
                    score=0.88,
                    metadata={
                        "document_id": "doc-1",
                        "chunk_index": 1,
                        "source_filename": "handbook.pdf",
                        "text": "second chunk",
                    },
                ),
                SimpleNamespace(
                    id="chunk-1",
                    score=0.66,
                    metadata={
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "source_filename": "handbook.pdf",
                        "text": "first chunk",
                    },
                ),
            ]
        )

        results = store.query([0.9, 0.1, 0.3], limit=2)

        mock_index.query.assert_called_once_with(
            vector=[0.9, 0.1, 0.3],
            top_k=2,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        assert len(results) == 2
        assert results[0].chunk_id == "chunk-2"
        assert results[0].distance == pytest.approx(0.12)
        assert results[1].chunk_id == "chunk-1"
        assert results[1].distance == pytest.approx(0.34)

    def test_query_filters_by_document_ids(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        mock_index.query.return_value = SimpleNamespace(matches=[])

        store.query([0.1, 0.2, 0.3], limit=5, document_ids=["doc-a", "doc-b"])

        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            include_metadata=True,
            namespace=NAMESPACE,
            filter={"document_id": {"$in": ["doc-a", "doc-b"]}},
        )

    def test_query_returns_empty_list_when_no_matches(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        mock_index.query.return_value = {"matches": []}

        assert store.query([0.1, 0.2], limit=5) == []


class TestPineconeVectorStoreDelete:
    def test_delete_document_filters_by_document_id(
        self,
        store: PineconeVectorStore,
        mock_index: MagicMock,
    ) -> None:
        store.delete_document("doc-123")

        mock_index.delete.assert_called_once_with(
            filter={"document_id": {"$eq": "doc-123"}},
            namespace=NAMESPACE,
        )


class TestPineconeVectorStoreDimensionValidation:
    def test_upsert_rejects_wrong_dimension(
        self,
        mock_client: MagicMock,
        mock_index: MagicMock,
    ) -> None:
        store = PineconeVectorStore(
            client=mock_client,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
            expected_dimension=3,
        )

        with pytest.raises(PineconeDimensionMismatchError, match="expected dimension 3"):
            store.upsert([_record(embedding=[0.1, 0.2])])

        mock_index.upsert.assert_not_called()

    def test_query_rejects_wrong_dimension(
        self,
        mock_client: MagicMock,
        mock_index: MagicMock,
    ) -> None:
        store = PineconeVectorStore(
            client=mock_client,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
            expected_dimension=3,
        )

        with pytest.raises(PineconeDimensionMismatchError, match="query embedding"):
            store.query([0.1, 0.2], limit=1)

        mock_index.query.assert_not_called()


class TestPineconeVectorStoreDependencyInjection:
    def test_concrete_store_is_substitutable_as_base_vector_store(
        self,
        mock_client: MagicMock,
    ) -> None:
        vector_store: BaseVectorStore = PineconeVectorStore(
            client=mock_client,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )

        vector_store.upsert([_record()])

        mock_client.Index.return_value.upsert.assert_called_once()
