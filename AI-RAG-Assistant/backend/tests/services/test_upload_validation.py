import pytest

from app.config import DOCUMENT_CONTENT_TYPE_IMAGE, DOCUMENT_CONTENT_TYPE_TEXT
from app.services.document_storage import UnsupportedDocumentType, classify_upload_extension


class TestClassifyUploadExtension:
    def test_classifies_text_documents(self) -> None:
        assert classify_upload_extension("notes.txt") == (".txt", DOCUMENT_CONTENT_TYPE_TEXT)
        assert classify_upload_extension("report.PDF") == (".pdf", DOCUMENT_CONTENT_TYPE_TEXT)

    def test_classifies_images(self) -> None:
        assert classify_upload_extension("photo.jpg") == (".jpg", DOCUMENT_CONTENT_TYPE_IMAGE)
        assert classify_upload_extension("scan.JPEG") == (".jpeg", DOCUMENT_CONTENT_TYPE_IMAGE)
        assert classify_upload_extension("slide.png") == (".png", DOCUMENT_CONTENT_TYPE_IMAGE)
        assert classify_upload_extension("photo.webp") == (".webp", DOCUMENT_CONTENT_TYPE_IMAGE)

    def test_rejects_unknown_extensions(self) -> None:
        with pytest.raises(UnsupportedDocumentType, match="Unsupported file type"):
            classify_upload_extension("archive.zip")
