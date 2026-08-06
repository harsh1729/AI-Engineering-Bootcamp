from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_ingestion import DocumentIndexingResult

client = TestClient(app)

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class TestUploadDocument:
    def test_upload_saves_and_indexes_document(
        self,
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
            "app.routers.documents.save_uploaded_document",
            lambda file: DOCUMENT_ID,
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: None,
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file: None,
        )

        with patch(
            "app.routers.documents.build_document_ingestion_service",
            return_value=mock_ingestion,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
            )

        assert response.status_code == 200
        assert response.json() == {
            "document_id": DOCUMENT_ID,
            "filename": "notes.txt",
            "status": "processed",
            "indexed_chunk_count": 4,
        }
        mock_ingestion.index_document.assert_called_once_with(
            DOCUMENT_ID,
            chunking_strategy="recursive",
        )

    def test_upload_returns_500_when_ingestion_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_ingestion = MagicMock()
        mock_ingestion.index_document.side_effect = RuntimeError("embed failed")

        monkeypatch.setattr(
            "app.routers.documents.save_uploaded_document",
            lambda file: DOCUMENT_ID,
        )
        monkeypatch.setattr(
            "app.routers.documents.validate_upload_extension",
            lambda filename: None,
        )
        monkeypatch.setattr(
            "app.routers.documents.enforce_upload_size_limit",
            lambda file: None,
        )

        with patch(
            "app.routers.documents.build_document_ingestion_service",
            return_value=mock_ingestion,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("notes.txt", BytesIO(b"hello world"), "text/plain")},
            )

        assert response.status_code == 500
        assert response.json()["detail"] == "Document processing failed."
