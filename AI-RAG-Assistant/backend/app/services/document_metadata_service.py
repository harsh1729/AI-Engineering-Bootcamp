import asyncio
import uuid

from app.database.database import AsyncSessionLocal
from app.database.repositories.document_repository import DocumentRepository
from app.models.document import DocumentMetadata
from app.models.rag_config import RagOptions
from app.services.document_exceptions import DocumentNotFoundError
from app.services.document_storage import _validate_document_id, resolve_document_path


async def save_indexed_document(
    repository: DocumentRepository,
    *,
    document_id: str,
    filename: str,
    rag_options: RagOptions,
    chunk_count: int,
    content_type: str = "text",
    user_id: uuid.UUID | None = None,
    guest_id: str | None = None,
) -> None:
    await repository.create(
        document_id=uuid.UUID(document_id),
        filename=filename,
        chunking_strategy=rag_options.chunking_strategy.value,
        embedding_provider=rag_options.embedding_provider.value,
        vector_db=rag_options.vector_store.value,
        chunk_count=chunk_count,
        content_type=content_type,
        user_id=user_id,
        guest_id=None if user_id is not None else guest_id,
    )


async def fetch_document_metadata(
    repository: DocumentRepository,
    document_id: str,
) -> DocumentMetadata:
    _validate_document_id(document_id)

    record = await repository.get_by_id(uuid.UUID(document_id))
    if record is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    stored_path = resolve_document_path(document_id)
    return DocumentMetadata(
        document_id=document_id,
        original_filename=record.filename,
        stored_filename=stored_path.name,
    )


def get_document_metadata(document_id: str) -> DocumentMetadata:
    """Load upload metadata from PostgreSQL for sync pipeline callers."""

    async def _load() -> DocumentMetadata:
        async with AsyncSessionLocal() as session:
            return await fetch_document_metadata(DocumentRepository(session), document_id)

    return asyncio.run(_load())
