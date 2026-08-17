import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.document import DocumentMetadata
from app.models.rag_config import RagOptions
from app.services.document_exceptions import DocumentNotFoundError
from app.services.document_metadata_service import (
    fetch_document_metadata,
    get_document_metadata,
    save_indexed_document,
)


@pytest.mark.asyncio
async def test_save_indexed_document_persists_with_explicit_id() -> None:
    repository = MagicMock()
    repository.create = AsyncMock()
    document_id = "11111111-2222-3333-4444-555555555555"
    options = RagOptions()

    await save_indexed_document(
        repository,
        document_id=document_id,
        filename="handbook.pdf",
        rag_options=options,
        chunk_count=5,
    )

    repository.create.assert_awaited_once_with(
        document_id=uuid.UUID(document_id),
        filename="handbook.pdf",
        chunking_strategy=options.chunking_strategy.value,
        embedding_provider=options.embedding_provider.value,
        vector_db=options.vector_store.value,
        chunk_count=5,
        content_type="text",
        user_id=None,
        guest_id=None,
    )


@pytest.mark.asyncio
async def test_fetch_document_metadata_loads_from_repository(
    tmp_path, monkeypatch
) -> None:
    from app.services import document_storage

    documents_dir = tmp_path / "files"
    documents_dir.mkdir()
    monkeypatch.setattr(document_storage, "DOCUMENTS_DIR", documents_dir)

    document_id = "11111111-2222-3333-4444-555555555555"
    (documents_dir / f"{document_id}.pdf").write_bytes(b"pdf")

    record = MagicMock()
    record.filename = "handbook.pdf"
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=record)

    metadata = await fetch_document_metadata(repository, document_id)

    assert metadata == DocumentMetadata(
        document_id=document_id,
        original_filename="handbook.pdf",
        stored_filename=f"{document_id}.pdf",
    )


@pytest.mark.asyncio
async def test_fetch_document_metadata_raises_when_missing() -> None:
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(DocumentNotFoundError):
        await fetch_document_metadata(
            repository,
            "11111111-2222-3333-4444-555555555555",
        )


def test_get_document_metadata_uses_async_loader() -> None:
    expected = DocumentMetadata(
        document_id="11111111-2222-3333-4444-555555555555",
        original_filename="notes.txt",
        stored_filename="11111111-2222-3333-4444-555555555555.txt",
    )

    with patch(
        "app.services.document_metadata_service.fetch_document_metadata",
        new=AsyncMock(return_value=expected),
    ):
        metadata = get_document_metadata(expected.document_id)

    assert metadata == expected
