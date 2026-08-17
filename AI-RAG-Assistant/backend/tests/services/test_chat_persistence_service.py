import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_sdk.enums import MessageRole

from app.database.models.chat_session import Chat
from app.models.chat import ChatMessage, ChatRequest
from app.services.chat_persistence_exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
)
from app.services.chat_persistence_service import (
    ChatPersistenceService,
    extract_latest_user_message,
    title_from_message,
)


def _chat(
    chat_id: uuid.UUID | None = None,
    guest_id: str = "guest-1",
    provider: str = "openai",
) -> Chat:
    return Chat(
        id=chat_id or uuid.uuid4(),
        guest_id=guest_id,
        provider=provider,
        user_id=None,
        title="Existing chat",
    )


def test_extract_latest_user_message_returns_last_user_message() -> None:
    request = ChatRequest(
        guest_id="guest-1",
        messages=[
            ChatMessage(role=MessageRole.USER, content="First"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Reply"),
            ChatMessage(role=MessageRole.USER, content="Second"),
        ],
    )

    assert extract_latest_user_message(request) == "Second"


def test_extract_latest_user_message_requires_user_message() -> None:
    request = ChatRequest(
        guest_id="guest-1",
        messages=[ChatMessage(role=MessageRole.ASSISTANT, content="Reply")],
    )

    with pytest.raises(ValueError, match="At least one user message"):
        extract_latest_user_message(request)


def test_title_from_message_truncates_long_content() -> None:
    long_content = "a" * 300

    title = title_from_message(long_content)

    assert len(title) == 255
    assert title.endswith("…")


def test_title_from_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="Message content is required"):
        title_from_message("   ")


@pytest.mark.asyncio
async def test_get_or_create_chat_returns_existing_chat_for_guest() -> None:
    chat = _chat()
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    message_repository = MagicMock()
    service = ChatPersistenceService(chat_repository, message_repository)

    resolved = await service.get_or_create_chat(
        guest_id="guest-1",
        provider="openai",
        chat_id=chat.id,
    )

    assert resolved is chat


@pytest.mark.asyncio
async def test_get_or_create_chat_returns_existing_chat_for_user() -> None:
    user_id = uuid.uuid4()
    chat = _chat()
    chat.guest_id = None
    chat.user_id = user_id
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    service = ChatPersistenceService(chat_repository, MagicMock())

    resolved = await service.get_or_create_chat(
        user_id=user_id,
        provider="openai",
        chat_id=chat.id,
    )

    assert resolved is chat


@pytest.mark.asyncio
async def test_get_or_create_chat_raises_when_chat_missing() -> None:
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=None)
    service = ChatPersistenceService(chat_repository, MagicMock())

    with pytest.raises(ChatNotFoundError):
        await service.get_or_create_chat(
            guest_id="guest-1",
            provider="openai",
            chat_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_get_or_create_chat_raises_when_guest_mismatch() -> None:
    chat = _chat(guest_id="other-guest")
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    service = ChatPersistenceService(chat_repository, MagicMock())

    with pytest.raises(ChatAccessDeniedError):
        await service.get_or_create_chat(
            guest_id="guest-1",
            provider="openai",
            chat_id=chat.id,
        )


@pytest.mark.asyncio
async def test_get_or_create_chat_allows_logged_in_user_to_access_guest_chat() -> None:
    user_id = uuid.uuid4()
    chat = _chat(guest_id="guest-1")
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    service = ChatPersistenceService(chat_repository, MagicMock())

    resolved = await service.get_or_create_chat(
        user_id=user_id,
        guest_id="guest-1",
        provider="openai",
        chat_id=chat.id,
    )

    assert resolved is chat


@pytest.mark.asyncio
async def test_get_or_create_chat_denies_logged_in_user_for_foreign_guest_chat() -> None:
    user_id = uuid.uuid4()
    chat = _chat(guest_id="other-guest")
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    service = ChatPersistenceService(chat_repository, MagicMock())

    with pytest.raises(ChatAccessDeniedError):
        await service.get_or_create_chat(
            user_id=user_id,
            guest_id="guest-1",
            provider="openai",
            chat_id=chat.id,
        )


@pytest.mark.asyncio
async def test_get_or_create_chat_creates_user_chat_without_guest_id() -> None:
    user_id = uuid.uuid4()
    created = _chat()
    created.guest_id = None
    created.user_id = user_id
    chat_repository = MagicMock()
    chat_repository.create = AsyncMock(return_value=created)
    service = ChatPersistenceService(chat_repository, MagicMock())

    resolved = await service.get_or_create_chat(
        user_id=user_id,
        guest_id="guest-1",
        provider="openai",
        first_message="Hello there",
    )

    assert resolved is created
    chat_repository.create.assert_awaited_once_with(
        user_id=user_id,
        guest_id=None,
        provider="openai",
        title="Hello there",
    )


@pytest.mark.asyncio
async def test_get_or_create_chat_raises_when_provider_mismatch() -> None:
    chat = _chat(provider="claude")
    chat_repository = MagicMock()
    chat_repository.get_by_id = AsyncMock(return_value=chat)
    service = ChatPersistenceService(chat_repository, MagicMock())

    with pytest.raises(ChatAccessDeniedError, match="does not belong to provider"):
        await service.get_or_create_chat(
            guest_id="guest-1",
            provider="openai",
            chat_id=chat.id,
        )


@pytest.mark.asyncio
async def test_get_or_create_chat_creates_on_first_message_when_chat_id_missing() -> None:
    created = _chat()
    chat_repository = MagicMock()
    chat_repository.create = AsyncMock(return_value=created)
    service = ChatPersistenceService(chat_repository, MagicMock())

    resolved = await service.get_or_create_chat(
        guest_id="guest-1",
        provider="openai",
        first_message="Hello there",
    )

    assert resolved is created
    chat_repository.create.assert_awaited_once_with(
        user_id=None,
        guest_id="guest-1",
        provider="openai",
        title="Hello there",
    )


@pytest.mark.asyncio
async def test_get_or_create_chat_requires_first_message_for_new_chat() -> None:
    service = ChatPersistenceService(MagicMock(), MagicMock())

    with pytest.raises(ValueError, match="first_message is required"):
        await service.get_or_create_chat(
            guest_id="guest-1",
            provider="openai",
        )


@pytest.mark.asyncio
async def test_append_user_message_persists_and_touches_chat() -> None:
    chat = _chat()
    chat_repository = MagicMock()
    chat_repository.touch = AsyncMock()
    message_repository = MagicMock()
    message_repository.create = AsyncMock()
    service = ChatPersistenceService(chat_repository, message_repository)

    await service.append_user_message(chat, "Hello")

    message_repository.create.assert_awaited_once_with(
        chat_id=chat.id,
        role="user",
        content="Hello",
    )
    chat_repository.touch.assert_awaited_once_with(chat)


@pytest.mark.asyncio
async def test_append_assistant_message_persists_provider_metadata() -> None:
    chat = _chat()
    chat_repository = MagicMock()
    chat_repository.touch = AsyncMock()
    message_repository = MagicMock()
    message_repository.create = AsyncMock()
    service = ChatPersistenceService(chat_repository, message_repository)

    await service.append_assistant_message(
        chat,
        "Hi",
        provider="openai",
        model="gpt-4o-mini",
    )

    message_repository.create.assert_awaited_once_with(
        chat_id=chat.id,
        role="assistant",
        content="Hi",
        provider="openai",
        model="gpt-4o-mini",
    )
