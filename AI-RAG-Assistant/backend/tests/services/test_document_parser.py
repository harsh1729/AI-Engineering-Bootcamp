import pytest

from app.models.document import ParsedDocument
from app.services import document_storage
from app.services.document_parser import DocumentParser


@pytest.fixture
def documents_dir(tmp_path, monkeypatch):
    documents_dir = tmp_path / "files"
    documents_dir.mkdir()
    monkeypatch.setattr(document_storage, "DOCUMENTS_DIR", documents_dir)
    return documents_dir


class TestDocumentParserFilename:
    def test_parse_returns_original_uploaded_filename_when_provided(
        self, documents_dir
    ) -> None:
        document_id = "11111111-2222-3333-4444-555555555555"
        stored_filename = f"{document_id}.txt"
        (documents_dir / stored_filename).write_text("parsed text", encoding="utf-8")

        parsed = DocumentParser().parse(
            document_id,
            original_filename="handbook.pdf",
        )

        assert parsed == ParsedDocument(
            document_id=document_id,
            filename="handbook.pdf",
            extracted_text="parsed text",
        )
