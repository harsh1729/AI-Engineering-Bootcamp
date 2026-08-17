import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.database.repositories.chat_repository import ChatRepository
from app.database.repositories.document_repository import DocumentRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.repositories.user_repository import UserRepository


@pytest.fixture
def session() -> MagicMock:
    mock = MagicMock()
    mock.add = MagicMock()
    mock.flush = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.execute = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_user_repository_create(session: MagicMock) -> None:
    repo = UserRepository(session)

    user = await repo.create(
        name="Demo User",
        email="demo@example.com",
        password_hash="hashed",
    )

    assert user.name == "Demo User"
    assert user.email == "demo@example.com"
    session.add.assert_called_once_with(user)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_repository_create_for_user(session: MagicMock) -> None:
    repo = ChatRepository(session)
    user_id = uuid.uuid4()

    chat = await repo.create(user_id=user_id, title="First chat")

    assert chat.user_id == user_id
    assert chat.guest_id is None
    assert chat.title == "First chat"
    session.add.assert_called_once_with(chat)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_repository_create_for_guest(session: MagicMock) -> None:
    repo = ChatRepository(session)
    guest_id = "11111111-2222-3333-4444-555555555555"

    chat = await repo.create(guest_id=guest_id, provider="openai", title="Guest chat")

    assert chat.user_id is None
    assert chat.guest_id == guest_id
    assert chat.provider == "openai"
    assert chat.title == "Guest chat"
    session.add.assert_called_once_with(chat)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_repository_create_requires_owner(session: MagicMock) -> None:
    repo = ChatRepository(session)

    with pytest.raises(ValueError, match="Either user_id or guest_id"):
        await repo.create(title="Orphan chat")


@pytest.mark.asyncio
async def test_message_repository_create(session: MagicMock) -> None:
    repo = MessageRepository(session)
    chat_id = uuid.uuid4()

    message = await repo.create(
        chat_id=chat_id,
        role="user",
        content="Hello",
        provider="openai",
        model="gpt-4o-mini",
    )

    assert message.chat_id == chat_id
    assert message.role == "user"
    assert message.content == "Hello"
    session.add.assert_called_once_with(message)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_document_repository_create(session: MagicMock) -> None:
    repo = DocumentRepository(session)
    document_id = uuid.uuid4()

    document = await repo.create(
        document_id=document_id,
        filename="handbook.pdf",
        chunking_strategy="recursive",
        embedding_provider="openai",
        vector_db="chroma",
        chunk_count=12,
        guest_id="guest-123",
    )

    assert document.id == document_id
    assert document.filename == "handbook.pdf"
    assert document.chunk_count == 12
    assert document.guest_id == "guest-123"
    assert document.user_id is None
    session.add.assert_called_once_with(document)
    session.flush.assert_awaited_once()
