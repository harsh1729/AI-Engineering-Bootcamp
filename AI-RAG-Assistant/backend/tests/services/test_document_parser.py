import json

import pytest

from app.models.document import DocumentMetadata, ParsedDocument
from app.services import document_storage
from app.services.document_parser import DocumentParser


@pytest.fixture
def storage_dirs(tmp_path, monkeypatch):
    documents_dir = tmp_path / "files"
    metadata_dir = tmp_path / "metadata"
    documents_dir.mkdir()
    metadata_dir.mkdir()
    monkeypatch.setattr(document_storage, "DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(document_storage, "METADATA_DIR", metadata_dir)
    return documents_dir, metadata_dir


class TestDocumentParserFilename:
    def test_parse_returns_original_uploaded_filename(
        self, storage_dirs
    ) -> None:
        documents_dir, metadata_dir = storage_dirs
        document_id = "11111111-2222-3333-4444-555555555555"
        stored_filename = f"{document_id}.txt"
        (documents_dir / stored_filename).write_text("parsed text", encoding="utf-8")
        (metadata_dir / f"{document_id}.json").write_text(
            json.dumps(
                DocumentMetadata(
                    document_id=document_id,
                    original_filename="handbook.pdf",
                    stored_filename=stored_filename,
                ).model_dump()
            ),
            encoding="utf-8",
        )

        parsed = DocumentParser().parse(document_id)

        assert parsed == ParsedDocument(
            document_id=document_id,
            filename="handbook.pdf",
            extracted_text="parsed text",
        )
