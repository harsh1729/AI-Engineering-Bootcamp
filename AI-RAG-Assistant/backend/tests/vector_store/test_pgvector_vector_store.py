from unittest.mock import MagicMock

import pytest

from app.services.document_exceptions import VectorDimensionMismatchError
from app.embeddings.embedding_models import EmbeddingVector
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.pgvector_vector_store import PgVectorVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

TABLE_NAME = "doc_pg_embeddings_openai"
MODEL = "text-embedding-3-small"


def _record(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "11111111-2222-3333-4444-555555555555",
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
def mock_connection() -> MagicMock:
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    return connection


@pytest.fixture
def mock_engine(mock_connection: MagicMock) -> MagicMock:
    engine = MagicMock()
    engine.begin.return_value = mock_connection
    return engine


@pytest.fixture
def store(mock_engine: MagicMock) -> PgVectorVectorStore:
    return PgVectorVectorStore(
        table_name=TABLE_NAME,
        engine=mock_engine,
    )


class TestPgVectorVectorStoreInit:
    def test_rejects_unknown_table_name(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="Invalid pgvector table name"):
            PgVectorVectorStore(table_name="unknown_table", engine=mock_engine)

    def test_requires_table_name(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="table_name is required"):
            PgVectorVectorStore(table_name="", engine=mock_engine)


class TestPgVectorVectorStoreUpsert:
    def test_upsert_executes_insert_for_each_record(
        self,
        store: PgVectorVectorStore,
        mock_connection: MagicMock,
    ) -> None:
        records = [
            _record(chunk_id="chunk-a", chunk_index=0, text="first"),
            _record(
                chunk_id="chunk-b",
                chunk_index=1,
                text="second",
                embedding=[0.4, 0.5, 0.6],
            ),
        ]

        store.upsert(records)

        assert mock_connection.execute.call_count == 2
        first_call = mock_connection.execute.call_args_list[0]
        assert first_call.args[1]["chunk_id"] == "chunk-a"
        assert first_call.args[1]["embedding"] == "[0.1,0.2,0.3]"
        second_call = mock_connection.execute.call_args_list[1]
        assert second_call.args[1]["chunk_id"] == "chunk-b"
        assert second_call.args[1]["embedding"] == "[0.4,0.5,0.6]"

    def test_empty_upsert_is_a_no_op(
        self,
        store: PgVectorVectorStore,
        mock_connection: MagicMock,
    ) -> None:
        store.upsert([])

        mock_connection.execute.assert_not_called()


class TestPgVectorVectorStoreQuery:
    def test_query_maps_rows_to_distance_ordered_results(
        self,
        store: PgVectorVectorStore,
        mock_connection: MagicMock,
    ) -> None:
        mock_connection.execute.return_value.mappings.return_value.all.return_value = [
            {
                "chunk_id": "chunk-2",
                "document_id": "11111111-2222-3333-4444-555555555555",
                "chunk_index": 1,
                "source_filename": "handbook.pdf",
                "text": "second chunk",
                "distance": 0.12,
            },
            {
                "chunk_id": "chunk-1",
                "document_id": "11111111-2222-3333-4444-555555555555",
                "chunk_index": 0,
                "source_filename": "handbook.pdf",
                "text": "first chunk",
                "distance": 0.34,
            },
        ]

        results = store.query([0.9, 0.1, 0.3], limit=2)

        assert len(results) == 2
        assert results[0] == VectorQueryResult(
            chunk_id="chunk-2",
            document_id="11111111-2222-3333-4444-555555555555",
            chunk_index=1,
            text="second chunk",
            source_filename="handbook.pdf",
            distance=0.12,
        )
        params = mock_connection.execute.call_args.args[1]
        assert params["query_embedding"] == "[0.9,0.1,0.3]"
        assert params["limit"] == 2
        assert "document_ids" not in params

    def test_query_filters_by_document_ids(
        self,
        store: PgVectorVectorStore,
        mock_connection: MagicMock,
    ) -> None:
        mock_connection.execute.return_value.mappings.return_value.all.return_value = []

        store.query(
            [0.1, 0.2, 0.3],
            limit=5,
            document_ids=["doc-a", "doc-b"],
        )

        params = mock_connection.execute.call_args.args[1]
        assert params["document_ids"] == ["doc-a", "doc-b"]
        sql = str(mock_connection.execute.call_args.args[0])
        assert "ANY(CAST(:document_ids AS uuid[]))" in sql


class TestPgVectorVectorStoreDelete:
    def test_delete_document_executes_delete_by_document_id(
        self,
        store: PgVectorVectorStore,
        mock_connection: MagicMock,
    ) -> None:
        store.delete_document("11111111-2222-3333-4444-555555555555")

        params = mock_connection.execute.call_args.args[1]
        assert params["document_id"] == "11111111-2222-3333-4444-555555555555"


class TestPgVectorVectorStoreDimensionValidation:
    def test_upsert_rejects_wrong_dimension(
        self,
        mock_engine: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        store = PgVectorVectorStore(
            table_name=TABLE_NAME,
            expected_dimension=3,
            engine=mock_engine,
        )

        with pytest.raises(VectorDimensionMismatchError, match="expected dimension 3"):
            store.upsert([_record(embedding=[0.1, 0.2])])

        mock_connection.execute.assert_not_called()

    def test_query_rejects_wrong_dimension(
        self,
        mock_engine: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        store = PgVectorVectorStore(
            table_name=TABLE_NAME,
            expected_dimension=3,
            engine=mock_engine,
        )

        with pytest.raises(VectorDimensionMismatchError, match="query embedding"):
            store.query([0.1, 0.2], limit=1)

        mock_connection.execute.assert_not_called()


class TestPgVectorVectorStoreDependencyInjection:
    def test_concrete_store_is_substitutable_as_base_vector_store(
        self,
        mock_engine: MagicMock,
    ) -> None:
        vector_store: BaseVectorStore = PgVectorVectorStore(
            table_name=TABLE_NAME,
            engine=mock_engine,
        )

        vector_store.upsert([_record()])

        mock_engine.begin.return_value.execute.assert_called_once()
