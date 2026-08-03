from unittest.mock import MagicMock, create_autospec

import pytest

from app.embeddings.embedding_models import EmbeddingVector
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.chroma_vector_store import ChromaVectorStore, _COSINE_CONFIGURATION
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

COLLECTION_NAME = "test-collection"
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
def mock_collection() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_client(mock_collection: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_collection
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> ChromaVectorStore:
    return ChromaVectorStore(client=mock_client, collection_name=COLLECTION_NAME)


class TestChromaVectorStoreCollection:
    def test_collection_is_created_with_cosine_distance(
        self, mock_client: MagicMock, mock_collection: MagicMock
    ) -> None:
        store = ChromaVectorStore(client=mock_client, collection_name=COLLECTION_NAME)

        mock_client.get_or_create_collection.assert_called_once_with(
            name=COLLECTION_NAME,
            configuration=_COSINE_CONFIGURATION,
        )


class TestChromaVectorStoreUpsert:
    def test_upsert_maps_ids_documents_embeddings_and_metadata(
        self, store: ChromaVectorStore, mock_collection: MagicMock
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

        mock_collection.upsert.assert_called_once_with(
            ids=["chunk-a", "chunk-b"],
            documents=["first", "second"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            metadatas=[
                {"document_id": "doc-a", "chunk_index": 0, "source_filename": "handbook.pdf"},
                {"document_id": "doc-a", "chunk_index": 1, "source_filename": "handbook.pdf"},
            ],
        )

    def test_upsert_stores_source_filename_metadata(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        store.upsert([_record(source_filename="annual-report.pdf")])

        metadata = mock_collection.upsert.call_args.kwargs["metadatas"][0]
        assert metadata["source_filename"] == "annual-report.pdf"

    def test_empty_upsert_is_a_no_op(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        store.upsert([])

        mock_collection.upsert.assert_not_called()


class TestChromaVectorStoreQuery:
    def test_query_maps_ids_texts_metadata_and_distances(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "ids": [["chunk-2", "chunk-1"]],
            "documents": [["second chunk", "first chunk"]],
            "metadatas": [[
                {"document_id": "doc-1", "chunk_index": 1, "source_filename": "handbook.pdf"},
                {"document_id": "doc-1", "chunk_index": 0, "source_filename": "handbook.pdf"},
            ]],
            "distances": [[0.12, 0.34]],
        }

        results = store.query([0.9, 0.1, 0.3], limit=2)

        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.9, 0.1, 0.3]],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        assert results == [
            VectorQueryResult(
                chunk_id="chunk-2",
                document_id="doc-1",
                chunk_index=1,
                text="second chunk",
                source_filename="handbook.pdf",
                distance=0.12,
            ),
            VectorQueryResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                chunk_index=0,
                text="first chunk",
                source_filename="handbook.pdf",
                distance=0.34,
            ),
        ]

    def test_query_maps_source_filename(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["first chunk"]],
            "metadatas": [[
                {
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "source_filename": "annual-report.pdf",
                }
            ]],
            "distances": [[0.2]],
        }

        results = store.query([0.1, 0.2, 0.3], limit=1)

        assert results[0].source_filename == "annual-report.pdf"

    def test_query_returns_empty_list_when_chroma_has_no_matches(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        assert store.query([0.1, 0.2], limit=5) == []

    def test_query_filters_by_document_ids(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        store.query([0.1, 0.2, 0.3], limit=5, document_ids=["doc-a", "doc-b"])

        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
            include=["documents", "metadatas", "distances"],
            where={"document_id": {"$in": ["doc-a", "doc-b"]}},
        )


class TestChromaVectorStoreDelete:
    def test_delete_document_filters_by_document_id(
        self, store: ChromaVectorStore, mock_collection: MagicMock
    ) -> None:
        store.delete_document("doc-123")

        mock_collection.delete.assert_called_once_with(where={"document_id": "doc-123"})


class TestBaseVectorStoreDependencyInjection:
    def test_store_can_be_used_through_base_vector_store_interface(self) -> None:
        provider = create_autospec(BaseVectorStore, instance=True)
        provider.query.return_value = expected = [
            VectorQueryResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                chunk_index=0,
                text="stored text",
                source_filename="handbook.pdf",
                distance=0.05,
            )
        ]

        results = provider.query([0.1, 0.2, 0.3], limit=1)

        provider.query.assert_called_once_with([0.1, 0.2, 0.3], limit=1)
        assert results is expected

    def test_concrete_store_is_substitutable_as_base_vector_store(
        self, mock_client: MagicMock
    ) -> None:
        vector_store: BaseVectorStore = ChromaVectorStore(
            client=mock_client,
            collection_name=COLLECTION_NAME,
        )

        vector_store.upsert([_record()])

        mock_client.get_or_create_collection.return_value.upsert.assert_called_once()
