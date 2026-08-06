from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.rag_config import ChunkingStrategy, RagOptions

client = TestClient(app)

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class TestRagOptionsCatalog:
    def test_returns_available_rag_options(self) -> None:
        response = client.get("/rag/options")

        assert response.status_code == 200
        payload = response.json()
        assert payload["defaults"]["chunking_strategy"] == "recursive"
        assert any(
            item["value"] == "recursive" for item in payload["chunking_strategies"]
        )
        assert any(item["value"] == "character" for item in payload["chunking_strategies"])
        assert any(item["value"] == "sentence" for item in payload["chunking_strategies"])
        assert any(item["value"] == "header_aware" for item in payload["chunking_strategies"])
        assert any(item["value"] == "openai" for item in payload["embedding_providers"])
        assert any(item["value"] == "voyage" for item in payload["embedding_providers"])
        assert any(item["value"] == "cohere" for item in payload["embedding_providers"])
        assert payload["embedding_models"]["openai"] == "text-embedding-3-small"
        assert payload["embedding_models"]["voyage"] == "voyage-4-lite"
        assert payload["embedding_models"]["cohere"] == "embed-v4.0"


class TestUploadWithRagOptions:
    def test_upload_builds_ingestion_service_from_rag_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_ingestion = MagicMock()
        mock_ingestion.index_document.return_value = MagicMock(
            indexed_chunk_count=3,
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
        ) as build_service:
            from io import BytesIO

            response = client.post(
                "/documents/upload",
                files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
                data={
                    "rag_options": '{"chunking_strategy":"character","embedding_provider":"openai","vector_store":"chroma"}',
                },
            )

        assert response.status_code == 200
        build_service.assert_called_once_with(
            RagOptions(
                chunking_strategy=ChunkingStrategy.CHARACTER,
            )
        )
        mock_ingestion.index_document.assert_called_once_with(
            DOCUMENT_ID,
            chunking_strategy="character",
        )
