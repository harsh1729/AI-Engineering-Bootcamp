import uuid
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import UploadFile

from app.models.document import DocumentMetadata
from app.services import document_storage
from app.services.document_exceptions import DocumentNotFoundError
from app.services.document_storage import (
    get_document_metadata,
    save_uploaded_document,
)


@pytest.fixture
def storage_dirs(tmp_path, monkeypatch):
    documents_dir = tmp_path / "files"
    metadata_dir = tmp_path / "metadata"
    monkeypatch.setattr(document_storage, "DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(document_storage, "METADATA_DIR", metadata_dir)
    return documents_dir, metadata_dir


def _upload(filename: str, content: bytes = b"hello world") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


class TestSaveUploadedDocumentMetadata:
    def test_duplicate_original_filenames_get_unique_storage_names(
        self, storage_dirs
    ) -> None:
        documents_dir, _ = storage_dirs

        first_id = save_uploaded_document(_upload("handbook.pdf", b"first"))
        second_id = save_uploaded_document(_upload("handbook.pdf", b"second"))

        assert first_id != second_id
        first_metadata = get_document_metadata(first_id)
        second_metadata = get_document_metadata(second_id)

        assert first_metadata.original_filename == "handbook.pdf"
        assert second_metadata.original_filename == "handbook.pdf"
        assert first_metadata.stored_filename == f"{first_id}.pdf"
        assert second_metadata.stored_filename == f"{second_id}.pdf"
        assert first_metadata.stored_filename != second_metadata.stored_filename
        assert (documents_dir / first_metadata.stored_filename).is_file()
        assert (documents_dir / second_metadata.stored_filename).is_file()

    def test_metadata_uses_basename_only_for_original_filename(
        self, storage_dirs
    ) -> None:
        document_id = save_uploaded_document(_upload("../../etc/handbook.pdf"))

        metadata = get_document_metadata(document_id)

        assert metadata.original_filename == "handbook.pdf"

    def test_metadata_survives_retrieval_by_document_id(
        self, storage_dirs
    ) -> None:
        document_id = save_uploaded_document(_upload("notes.txt", b"content"))

        metadata = get_document_metadata(document_id)

        assert metadata == DocumentMetadata(
            document_id=document_id,
            original_filename="notes.txt",
            stored_filename=f"{document_id}.txt",
        )

    def test_get_document_metadata_rejects_invalid_document_id(
        self, storage_dirs
    ) -> None:
        with pytest.raises(DocumentNotFoundError):
            get_document_metadata("../bad-id")

    def test_metadata_persistence_failure_removes_uploaded_file(
        self, storage_dirs
    ) -> None:
        documents_dir, _ = storage_dirs
        document_id = str(uuid.uuid4())

        with patch.object(
            document_storage,
            "_persist_metadata",
            side_effect=OSError("metadata write failed"),
        ):
            with patch.object(document_storage.uuid, "uuid4", return_value=uuid.UUID(document_id)):
                with pytest.raises(OSError, match="metadata write failed"):
                    save_uploaded_document(_upload("notes.txt"))

        assert not (documents_dir / f"{document_id}.txt").exists()
