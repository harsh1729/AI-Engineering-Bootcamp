import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.database.models.stored_document import Document
from app.services.document_access_service import DocumentAccessDeniedError, DocumentAccessService

USER_ID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
GUEST_ID = "guest-abc"
OTHER_GUEST_ID = "guest-xyz"
DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _document(*, user_id=None, guest_id=None) -> Document:
    document = MagicMock(spec=Document)
    document.id = DOCUMENT_ID
    document.user_id = user_id
    document.guest_id = guest_id
    return document


@pytest.fixture
def repository() -> MagicMock:
    mock = MagicMock()
    mock.list_by_guest = AsyncMock(return_value=[])
    mock.list_by_user = AsyncMock(return_value=[])
    mock.list_by_user_and_guest = AsyncMock(return_value=[])
    mock.get_by_id = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def service(repository: MagicMock) -> DocumentAccessService:
    return DocumentAccessService(repository)


@pytest.mark.asyncio
async def test_list_documents_guest_only(service: DocumentAccessService, repository: MagicMock) -> None:
    await service.list_documents(user_id=None, guest_id=GUEST_ID)

    repository.list_by_guest.assert_awaited_once_with(GUEST_ID)
    repository.list_by_user.assert_not_called()
    repository.list_by_user_and_guest.assert_not_called()


@pytest.mark.asyncio
async def test_list_documents_user_only(service: DocumentAccessService, repository: MagicMock) -> None:
    await service.list_documents(user_id=USER_ID, guest_id=None)

    repository.list_by_user.assert_awaited_once_with(USER_ID)
    repository.list_by_user_and_guest.assert_not_called()


@pytest.mark.asyncio
async def test_list_documents_user_and_guest_union(
    service: DocumentAccessService,
    repository: MagicMock,
) -> None:
    await service.list_documents(user_id=USER_ID, guest_id=GUEST_ID)

    repository.list_by_user_and_guest.assert_awaited_once_with(USER_ID, GUEST_ID)


@pytest.mark.asyncio
async def test_validate_document_ids_allows_owned_guest_document(
    service: DocumentAccessService,
    repository: MagicMock,
) -> None:
    repository.get_by_id.return_value = _document(guest_id=GUEST_ID)

    await service.validate_document_ids(
        [str(DOCUMENT_ID)],
        user_id=None,
        guest_id=GUEST_ID,
    )


@pytest.mark.asyncio
async def test_validate_document_ids_allows_user_or_guest_when_logged_in(
    service: DocumentAccessService,
    repository: MagicMock,
) -> None:
    repository.get_by_id.return_value = _document(guest_id=GUEST_ID)

    await service.validate_document_ids(
        [str(DOCUMENT_ID)],
        user_id=USER_ID,
        guest_id=GUEST_ID,
    )


@pytest.mark.asyncio
async def test_validate_document_ids_denies_other_guest(
    service: DocumentAccessService,
    repository: MagicMock,
) -> None:
    repository.get_by_id.return_value = _document(guest_id=OTHER_GUEST_ID)

    with pytest.raises(DocumentAccessDeniedError):
        await service.validate_document_ids(
            [str(DOCUMENT_ID)],
            user_id=None,
            guest_id=GUEST_ID,
        )


@pytest.mark.asyncio
async def test_validate_document_ids_denies_missing_document(
    service: DocumentAccessService,
    repository: MagicMock,
) -> None:
    repository.get_by_id.return_value = None

    with pytest.raises(DocumentAccessDeniedError):
        await service.validate_document_ids(
            [str(DOCUMENT_ID)],
            user_id=None,
            guest_id=GUEST_ID,
        )
