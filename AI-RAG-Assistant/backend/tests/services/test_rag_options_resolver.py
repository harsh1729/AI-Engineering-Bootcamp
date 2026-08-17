import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.database.models.stored_document import Document
from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)
from app.services.rag_options_resolver import (
    InvalidDocumentRagConfigError,
    MixedRagConfigError,
    RagOptionsResolverService,
)

DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER_DOCUMENT_ID = uuid.UUID("22222222-3333-4444-5555-666666666666")


def _document(
    document_id: uuid.UUID,
    *,
    chunking_strategy: str = "recursive",
    embedding_provider: str = "openai",
    vector_db: str = "chroma",
) -> Document:
    document = MagicMock(spec=Document)
    document.id = document_id
    document.chunking_strategy = chunking_strategy
    document.embedding_provider = embedding_provider
    document.vector_db = vector_db
    return document


@pytest.fixture
def repository() -> MagicMock:
    return MagicMock()


@pytest.fixture
def resolver(repository: MagicMock) -> RagOptionsResolverService:
    return RagOptionsResolverService(repository)


@pytest.mark.asyncio
async def test_resolve_rag_options_from_single_document(
    resolver: RagOptionsResolverService,
    repository: MagicMock,
) -> None:
    repository.get_by_id = AsyncMock(return_value=_document(DOCUMENT_ID))

    options = await resolver.resolve_rag_options_for_documents([str(DOCUMENT_ID)])

    assert options == RagOptions(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_provider=EmbeddingProviderType.OPENAI,
        vector_store=VectorStoreType.CHROMA,
    )


@pytest.mark.asyncio
async def test_resolve_rag_options_requires_matching_configs(
    resolver: RagOptionsResolverService,
    repository: MagicMock,
) -> None:
    async def get_by_id(document_id: uuid.UUID) -> Document | None:
        if document_id == DOCUMENT_ID:
            return _document(DOCUMENT_ID, chunking_strategy="recursive")
        if document_id == OTHER_DOCUMENT_ID:
            return _document(OTHER_DOCUMENT_ID, chunking_strategy="character")
        return None

    repository.get_by_id = AsyncMock(side_effect=get_by_id)

    with pytest.raises(MixedRagConfigError, match="different indexing settings"):
        await resolver.resolve_rag_options_for_documents(
            [str(DOCUMENT_ID), str(OTHER_DOCUMENT_ID)],
        )


@pytest.mark.asyncio
async def test_resolve_rag_options_allows_multiple_documents_with_same_config(
    resolver: RagOptionsResolverService,
    repository: MagicMock,
) -> None:
    repository.get_by_id = AsyncMock(return_value=_document(DOCUMENT_ID))

    options = await resolver.resolve_rag_options_for_documents(
        [str(DOCUMENT_ID), str(OTHER_DOCUMENT_ID)],
    )

    assert options.vector_store == VectorStoreType.CHROMA
    assert repository.get_by_id.await_count == 2


@pytest.mark.asyncio
async def test_resolve_rag_options_raises_when_document_missing(
    resolver: RagOptionsResolverService,
    repository: MagicMock,
) -> None:
    repository.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(InvalidDocumentRagConfigError, match="was not found"):
        await resolver.resolve_rag_options_for_documents([str(DOCUMENT_ID)])


@pytest.mark.asyncio
async def test_resolve_rag_options_raises_for_invalid_stored_config(
    resolver: RagOptionsResolverService,
    repository: MagicMock,
) -> None:
    repository.get_by_id = AsyncMock(
        return_value=_document(DOCUMENT_ID, vector_db="unknown-store"),
    )

    with pytest.raises(InvalidDocumentRagConfigError, match="invalid indexing configuration"):
        await resolver.resolve_rag_options_for_documents([str(DOCUMENT_ID)])
