import uuid
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.config import DOCUMENT_CONTENT_TYPE_IMAGE, DOCUMENT_CONTENT_TYPE_TEXT
from app.services import document_storage
from app.services.document_exceptions import DocumentNotFoundError
from app.services.document_storage import (
    resolve_document_path,
    save_uploaded_file,
)


@pytest.fixture
def documents_dir(tmp_path, monkeypatch):
    documents_dir = tmp_path / "files"
    monkeypatch.setattr(document_storage, "DOCUMENTS_DIR", documents_dir)
    return documents_dir


@pytest.fixture
def images_dir(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    monkeypatch.setattr(document_storage, "IMAGES_DIR", images_dir)
    return images_dir


def _upload(filename: str, content: bytes = b"hello world") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


class TestSaveUploadedDocument:
    def test_duplicate_original_filenames_get_unique_storage_names(
        self, documents_dir
    ) -> None:
        first_id, _ = save_uploaded_file(_upload("handbook.pdf", b"first"))
        second_id, _ = save_uploaded_file(_upload("handbook.pdf", b"second"))

        assert first_id != second_id
        assert (documents_dir / f"{first_id}.pdf").is_file()
        assert (documents_dir / f"{second_id}.pdf").is_file()
        assert (documents_dir / f"{first_id}.pdf").read_bytes() == b"first"
        assert (documents_dir / f"{second_id}.pdf").read_bytes() == b"second"

    def test_save_uses_basename_extension_for_stored_file(self, documents_dir) -> None:
        document_id, content_type = save_uploaded_file(_upload("../../etc/handbook.pdf"))

        stored_path = documents_dir / f"{document_id}.pdf"
        assert content_type == DOCUMENT_CONTENT_TYPE_TEXT
        assert stored_path.is_file()

    def test_resolve_document_path_finds_uploaded_file(self, documents_dir) -> None:
        document_id, _ = save_uploaded_file(_upload("notes.txt", b"content"))

        resolved = resolve_document_path(document_id)

        assert resolved == documents_dir / f"{document_id}.txt"
        assert resolved.read_text(encoding="utf-8") == "content"

    def test_resolve_document_path_rejects_invalid_document_id(
        self, documents_dir
    ) -> None:
        with pytest.raises(DocumentNotFoundError):
            resolve_document_path("../bad-id")

    def test_save_returns_valid_uuid(self, documents_dir) -> None:
        document_id, _ = save_uploaded_file(_upload("notes.txt"))

        parsed = uuid.UUID(document_id)
        assert str(parsed) == document_id


class TestSaveUploadedImage:
    def test_save_image_to_images_dir(self, images_dir) -> None:
        document_id, content_type = save_uploaded_file(_upload("scan.png", b"png-bytes"))

        assert content_type == DOCUMENT_CONTENT_TYPE_IMAGE
        stored_path = images_dir / f"{document_id}.png"
        assert stored_path.is_file()
        assert stored_path.read_bytes() == b"png-bytes"

    def test_resolve_document_path_finds_image_upload(self, images_dir) -> None:
        document_id, _ = save_uploaded_file(_upload("photo.jpg", b"jpeg-bytes"))

        resolved = resolve_document_path(document_id)

        assert resolved == images_dir / f"{document_id}.jpg"
