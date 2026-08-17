import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.auth.deps import get_optional_current_user
from app.database.deps import get_chat_repository, get_message_repository
from app.database.models.chat_session import Chat
from app.database.models.message import Message
from app.database.models.user import User
from app.main import app
from app.services.jwt_service import create_access_token

GUEST_ID = "11111111-2222-3333-4444-555555555555"
USER_ID = uuid.UUID("33333333-4444-5555-6666-777777777777")
PROVIDER = "openai"
CHAT_ID = uuid.UUID("22222222-3333-4444-5555-666666666666")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def chat_repository() -> MagicMock:
    return MagicMock()


@pytest.fixture
def message_repository() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(chat_repository: MagicMock, message_repository: MagicMock) -> TestClient:
    app.dependency_overrides[get_chat_repository] = lambda: chat_repository
    app.dependency_overrides[get_message_repository] = lambda: message_repository
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _chat() -> Chat:
    return Chat(
        id=CHAT_ID,
        guest_id=GUEST_ID,
        provider=PROVIDER,
        user_id=None,
        title="First chat",
        created_at=NOW,
        updated_at=NOW,
    )


def _message(role: str, content: str) -> Message:
    return Message(
        id=uuid.uuid4(),
        chat_id=CHAT_ID,
        role=role,
        content=content,
        provider="openai" if role == "assistant" else None,
        model="gpt-4o-mini" if role == "assistant" else None,
        created_at=NOW,
    )


class TestCreateChat:
    def test_create_chat_for_guest(
        self,
        client: TestClient,
        chat_repository: MagicMock,
    ) -> None:
        chat_repository.create = AsyncMock(return_value=_chat())

        response = client.post(
            "/chats",
            json={"guest_id": GUEST_ID, "provider": PROVIDER, "title": "First chat"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "id": str(CHAT_ID),
            "guest_id": GUEST_ID,
            "user_id": None,
            "provider": PROVIDER,
            "title": "First chat",
            "created_at": NOW.isoformat().replace("+00:00", "Z"),
            "updated_at": NOW.isoformat().replace("+00:00", "Z"),
        }
        chat_repository.create.assert_awaited_once_with(
            guest_id=GUEST_ID,
            provider=PROVIDER,
            title="First chat",
        )

    def test_create_chat_uses_default_title(
        self,
        client: TestClient,
        chat_repository: MagicMock,
    ) -> None:
        chat_repository.create = AsyncMock(return_value=_chat())

        response = client.post(
            "/chats",
            json={"guest_id": GUEST_ID, "provider": PROVIDER},
        )

        assert response.status_code == 200
        chat_repository.create.assert_awaited_once_with(
            guest_id=GUEST_ID,
            provider=PROVIDER,
            title="New chat",
        )


class TestListChats:
    def test_list_chats_for_guest_and_provider(
        self,
        client: TestClient,
        chat_repository: MagicMock,
    ) -> None:
        chat_repository.list_by_guest_and_provider = AsyncMock(return_value=[_chat()])

        response = client.get(
            "/chats",
            params={"guest_id": GUEST_ID, "provider": PROVIDER},
        )

        assert response.status_code == 200
        assert response.json() == {
            "chats": [
                {
                    "id": str(CHAT_ID),
                    "guest_id": GUEST_ID,
                    "user_id": None,
                    "provider": PROVIDER,
                    "title": "First chat",
                    "created_at": NOW.isoformat().replace("+00:00", "Z"),
                    "updated_at": NOW.isoformat().replace("+00:00", "Z"),
                }
            ]
        }
        chat_repository.list_by_guest_and_provider.assert_awaited_once_with(
            GUEST_ID,
            PROVIDER,
        )

    def test_list_chats_unions_guest_and_user_when_logged_in(
        self,
        client: TestClient,
        chat_repository: MagicMock,
    ) -> None:
        user = User(
            id=USER_ID,
            name="Demo User",
            email="demo@example.com",
            password_hash="hashed",
            is_approved=True,
            is_admin=False,
            is_enabled=True,
            created_at=NOW,
            updated_at=NOW,
            last_login_at=None,
        )
        app.dependency_overrides[get_optional_current_user] = lambda: user
        chat_repository.list_by_user_and_guest_and_provider = AsyncMock(
            return_value=[_chat()],
        )

        token = create_access_token(USER_ID)
        response = client.get(
            "/chats",
            params={"guest_id": GUEST_ID, "provider": PROVIDER},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        chat_repository.list_by_user_and_guest_and_provider.assert_awaited_once_with(
            USER_ID,
            GUEST_ID,
            PROVIDER,
        )
        app.dependency_overrides.pop(get_optional_current_user, None)


class TestListChatMessages:
    def test_list_messages_for_owned_chat(
        self,
        client: TestClient,
        chat_repository: MagicMock,
        message_repository: MagicMock,
    ) -> None:
        user_message = _message("user", "Hello")
        assistant_message = _message("assistant", "Hi there")
        chat_repository.get_by_id_for_guest = AsyncMock(return_value=_chat())
        message_repository.list_by_chat = AsyncMock(
            return_value=[user_message, assistant_message]
        )

        response = client.get(
            f"/chats/{CHAT_ID}/messages",
            params={"guest_id": GUEST_ID, "provider": PROVIDER},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["chat_id"] == str(CHAT_ID)
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == "Hi there"

    def test_list_messages_returns_404_for_foreign_chat(
        self,
        client: TestClient,
        chat_repository: MagicMock,
    ) -> None:
        chat_repository.get_by_id_for_guest = AsyncMock(return_value=None)

        response = client.get(
            f"/chats/{CHAT_ID}/messages",
            params={"guest_id": GUEST_ID, "provider": PROVIDER},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == f"Chat '{CHAT_ID}' was not found."
