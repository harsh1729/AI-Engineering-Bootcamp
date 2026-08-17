from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth.deps import get_optional_current_user
from app.database.deps import get_document_access_service, get_document_repository
from app.database.models.stored_document import Document
from app.main import app
from app.services.document_ingestion import DocumentIndexingResult

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"
GUEST_ID = "guest-123"
USER_ID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _document(*, user_id=None, guest_id=GUEST_ID) -> Document:
    document = MagicMock(spec=Document)
    document.id = uuid.UUID(DOCUMENT_ID)
    document.filename = "notes.txt"
    document.content_type = "text"
    document.chunking_strategy = "recursive"
    document.embedding_provider = "openai"
    document.vector_db = "chroma"
    document.chunk_count = 4
    document.user_id = user_id
    document.guest_id = guest_id
    return document


@pytest.fixture
def mock_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_access() -> MagicMock:
    mock = MagicMock()
    mock.list_documents = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def client(mock_repo: MagicMock, mock_access: MagicMock) -> TestClient:
    async def override_document_repository():
        yield mock_repo

    async def override_document_access_service():
        yield mock_access

    app.dependency_overrides[get_document_repository] = override_document_repository
    app.dependency_overrides[get_document_access_service] = override_document_access_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestUploadDocument:
    def test_upload_saves_and_indexes_document(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_ingestion = MagicMock()
        mock_ingestion.index_document.return_value = DocumentIndexingResult(
            document_id=DOCUMENT_ID,
            filename="notes.txt",
            indexed_chunk_count=4,
            embedding_model="text-embedding-3-small",
            usage=None,
        )

        monkeypatch.setattr(
            "app.routers.documents.save_uploaded_file",
            lambda file: (DOCUMENT_ID, "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: (".txt", "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file, content_type: None,
        )

        with patch(
            "app.routers.documents.build_document_ingestion_service",
            return_value=mock_ingestion,
        ), patch(
            "app.routers.documents.save_indexed_document",
            new=AsyncMock(),
        ) as save_metadata:
            response = client.post(
                "/documents/upload",
                data={"guest_id": GUEST_ID},
                files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
            )

        assert response.status_code == 200
        assert response.json() == {
            "document_id": DOCUMENT_ID,
            "filename": "notes.txt",
            "status": "processed",
            "indexed_chunk_count": 4,
        }
        save_metadata.assert_awaited_once()
        assert save_metadata.await_args.kwargs["guest_id"] == GUEST_ID
        assert save_metadata.await_args.kwargs["user_id"] is None
        assert save_metadata.await_args.kwargs["content_type"] == "text"
        mock_ingestion.index_document.assert_called_once_with(
            DOCUMENT_ID,
            chunking_strategy="recursive",
            original_filename="notes.txt",
        )

    def test_upload_requires_guest_id_when_unauthenticated(self, client: TestClient) -> None:
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "guest_id is required for unauthenticated requests."

    def test_upload_assigns_user_id_when_authenticated(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user = MagicMock()
        user.id = USER_ID
        app.dependency_overrides[get_optional_current_user] = lambda: user

        mock_ingestion = MagicMock()
        mock_ingestion.index_document.return_value = DocumentIndexingResult(
            document_id=DOCUMENT_ID,
            filename="notes.txt",
            indexed_chunk_count=4,
            embedding_model="text-embedding-3-small",
            usage=None,
        )

        monkeypatch.setattr(
            "app.routers.documents.save_uploaded_file",
            lambda file: (DOCUMENT_ID, "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: (".txt", "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file, content_type: None,
        )

        try:
            with patch(
                "app.routers.documents.build_document_ingestion_service",
                return_value=mock_ingestion,
            ), patch(
                "app.routers.documents.save_indexed_document",
                new=AsyncMock(),
            ) as save_metadata:
                response = client.post(
                    "/documents/upload",
                    data={"guest_id": GUEST_ID},
                    files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
                )
        finally:
            app.dependency_overrides.pop(get_optional_current_user, None)

        assert response.status_code == 200
        save_metadata.assert_awaited_once()
        assert save_metadata.await_args.kwargs["user_id"] == USER_ID
        assert save_metadata.await_args.kwargs["guest_id"] is None

    def test_upload_returns_500_when_ingestion_fails(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_ingestion = MagicMock()
        mock_ingestion.index_document.side_effect = RuntimeError("embed failed")

        monkeypatch.setattr(
            "app.routers.documents.save_uploaded_file",
            lambda file: (DOCUMENT_ID, "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: (".txt", "text"),
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file, content_type: None,
        )

        with patch(
            "app.routers.documents.build_document_ingestion_service",
            return_value=mock_ingestion,
        ):
            response = client.post(
                "/documents/upload",
                data={"guest_id": GUEST_ID},
                files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
            )

        assert response.status_code == 500
        assert response.json()["detail"] == "Document processing failed."


class TestUploadImage:
    def test_upload_image_persists_image_content_type(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_ingestion = MagicMock()
        mock_ingestion.index_document.return_value = DocumentIndexingResult(
            document_id=DOCUMENT_ID,
            filename="scan.png",
            indexed_chunk_count=1,
            embedding_model="text-embedding-3-small",
            usage=None,
        )

        monkeypatch.setattr(
            "app.routers.documents.save_uploaded_file",
            lambda file: (DOCUMENT_ID, "image"),
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: (".png", "image"),
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file, content_type: None,
        )

        with patch(
            "app.routers.documents.build_document_ingestion_service",
            return_value=mock_ingestion,
        ), patch(
            "app.routers.documents.save_indexed_document",
            new=AsyncMock(),
        ) as save_metadata:
            response = client.post(
                "/documents/upload",
                data={"guest_id": GUEST_ID},
                files={"file": ("scan.png", BytesIO(b"png-bytes"), "image/png")},
            )

        assert response.status_code == 200
        save_metadata.assert_awaited_once()
        assert save_metadata.await_args.kwargs["content_type"] == "image"
        mock_ingestion.index_document.assert_called_once_with(
            DOCUMENT_ID,
            chunking_strategy="recursive",
            original_filename="scan.png",
        )


class TestListDocuments:
    def test_list_documents_guest_only(
        self,
        client: TestClient,
        mock_access: MagicMock,
    ) -> None:
        document = _document(guest_id=GUEST_ID)
        mock_access.list_documents.return_value = [document]

        response = client.get("/documents", params={"guest_id": GUEST_ID})

        assert response.status_code == 200
        payload = response.json()
        assert len(payload["documents"]) == 1
        assert payload["documents"][0]["id"] == DOCUMENT_ID
        assert payload["documents"][0]["guest_id"] == GUEST_ID
        mock_access.list_documents.assert_awaited_once_with(
            user_id=None,
            guest_id=GUEST_ID,
        )

    def test_list_documents_union_when_authenticated(
        self,
        client: TestClient,
        mock_access: MagicMock,
    ) -> None:
        user = MagicMock()
        user.id = USER_ID
        app.dependency_overrides[get_optional_current_user] = lambda: user
        mock_access.list_documents.return_value = [_document(user_id=USER_ID, guest_id=None)]

        try:
            response = client.get("/documents", params={"guest_id": GUEST_ID})
        finally:
            app.dependency_overrides.pop(get_optional_current_user, None)

        assert response.status_code == 200
        mock_access.list_documents.assert_awaited_once_with(
            user_id=USER_ID,
            guest_id=GUEST_ID,
        )

    def test_list_documents_requires_guest_id_when_unauthenticated(
        self,
        client: TestClient,
    ) -> None:
        response = client.get("/documents")

        assert response.status_code == 400
        assert response.json()["detail"] == "guest_id is required for unauthenticated requests."
