import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import REGISTRATION_PENDING_MESSAGE
from app.database.deps import get_user_repository
from app.database.models.user import User
from app.main import app
from app.services.jwt_service import create_access_token

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
USER_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
ADMIN_ID = uuid.UUID("22222222-3333-4444-5555-666666666666")
PENDING_ID = uuid.UUID("33333333-4444-5555-6666-777777777777")


def _user(
    *,
    user_id: uuid.UUID = USER_ID,
    is_approved: bool = True,
    is_admin: bool = False,
    is_enabled: bool = True,
    email: str = "demo@example.com",
    name: str = "Demo User",
) -> User:
    return User(
        id=user_id,
        name=name,
        email=email,
        password_hash="hashed",
        is_approved=is_approved,
        is_admin=is_admin,
        is_enabled=is_enabled,
        created_at=NOW,
        updated_at=NOW,
        last_login_at=None,
    )


@pytest.fixture
def user_repository() -> MagicMock:
    mock = MagicMock()
    mock.get_by_email = AsyncMock(return_value=None)
    mock.create = AsyncMock(return_value=_user(is_approved=False))
    mock.update_last_login = AsyncMock(return_value=_user())
    mock.get_by_id = AsyncMock(return_value=_user())
    mock.list_pending_non_admin = AsyncMock(return_value=[])
    mock.list_approved_non_admin = AsyncMock(return_value=[])
    mock.approve = AsyncMock(return_value=_user())
    mock.set_enabled = AsyncMock(return_value=_user())
    return mock


@pytest.fixture
def client(user_repository: MagicMock) -> TestClient:
    app.dependency_overrides[get_user_repository] = lambda: user_repository
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestRegister:
    def test_register_creates_pending_user_without_token(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        response = client.post(
            "/auth/register",
            json={
                "name": "Demo User",
                "email": "demo@example.com",
                "password": "secret123",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"message": REGISTRATION_PENDING_MESSAGE}
        user_repository.create.assert_awaited_once()
        user_repository.update_last_login.assert_not_called()

    def test_register_returns_409_for_duplicate_email(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        user_repository.get_by_email = AsyncMock(return_value=_user())

        response = client.post(
            "/auth/register",
            json={
                "name": "Demo User",
                "email": "demo@example.com",
                "password": "secret123",
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestLogin:
    def test_login_returns_token_for_valid_credentials(
        self,
        client: TestClient,
        user_repository: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_repository.get_by_email = AsyncMock(return_value=_user())
        monkeypatch.setattr(
            "app.services.auth_service.verify_password",
            lambda _password, _hash: True,
        )

        response = client.post(
            "/auth/login",
            json={"email": "demo@example.com", "password": "secret123"},
        )

        assert response.status_code == 200
        assert response.json()["user"]["email"] == "demo@example.com"
        assert response.json()["user"]["is_admin"] is False
        user_repository.update_last_login.assert_awaited_once_with(USER_ID)

    def test_login_returns_401_for_invalid_credentials(
        self,
        client: TestClient,
        user_repository: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_repository.get_by_email = AsyncMock(return_value=_user())
        monkeypatch.setattr(
            "app.services.auth_service.verify_password",
            lambda _password, _hash: False,
        )

        response = client.post(
            "/auth/login",
            json={"email": "demo@example.com", "password": "wrong-password"},
        )

        assert response.status_code == 401

    def test_login_returns_403_for_unapproved_user(
        self,
        client: TestClient,
        user_repository: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_repository.get_by_email = AsyncMock(return_value=_user(is_approved=False))
        monkeypatch.setattr(
            "app.services.auth_service.verify_password",
            lambda _password, _hash: True,
        )

        response = client.post(
            "/auth/login",
            json={"email": "demo@example.com", "password": "secret123"},
        )

        assert response.status_code == 403

    def test_login_returns_403_for_disabled_user(
        self,
        client: TestClient,
        user_repository: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_repository.get_by_email = AsyncMock(
            return_value=_user(is_enabled=False),
        )
        monkeypatch.setattr(
            "app.services.auth_service.verify_password",
            lambda _password, _hash: True,
        )

        response = client.post(
            "/auth/login",
            json={"email": "demo@example.com", "password": "secret123"},
        )

        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()


class TestMe:
    def test_me_returns_current_user(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        token = create_access_token(USER_ID)

        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        payload = response.json()["user"]
        assert payload["email"] == "demo@example.com"
        assert payload["is_admin"] is False

    def test_me_returns_401_without_token(self, client: TestClient) -> None:
        response = client.get("/auth/me")

        assert response.status_code == 401


class TestAdminUsers:
    def test_list_users_requires_admin(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        token = create_access_token(USER_ID)

        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403

    def test_list_users_returns_pending_and_approved(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        pending = _user(
            user_id=PENDING_ID,
            is_approved=False,
            email="pending@example.com",
            name="Pending User",
        )
        approved = _user(email="approved@example.com", name="Approved User")
        user_repository.get_by_id = AsyncMock(return_value=_user(user_id=ADMIN_ID, is_admin=True))
        user_repository.list_pending_non_admin = AsyncMock(return_value=[pending])
        user_repository.list_approved_non_admin = AsyncMock(return_value=[approved])

        token = create_access_token(ADMIN_ID)
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert len(payload["pending"]) == 1
        assert payload["pending"][0]["email"] == "pending@example.com"
        assert len(payload["approved"]) == 1
        assert payload["approved"][0]["email"] == "approved@example.com"

    def test_approve_user_moves_user_to_approved(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        approved = _user(user_id=PENDING_ID, email="pending@example.com")
        user_repository.get_by_id = AsyncMock(
            side_effect=lambda user_id: (
                _user(user_id=ADMIN_ID, is_admin=True)
                if user_id == ADMIN_ID
                else _user(
                    user_id=PENDING_ID,
                    is_approved=False,
                    is_admin=False,
                )
            )
        )
        user_repository.approve = AsyncMock(return_value=approved)

        token = create_access_token(ADMIN_ID)
        response = client.post(
            f"/admin/users/{PENDING_ID}/approve",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["email"] == "pending@example.com"
        user_repository.approve.assert_awaited_once_with(PENDING_ID)

    def test_disable_user_sets_enabled_false(
        self,
        client: TestClient,
        user_repository: MagicMock,
    ) -> None:
        disabled = _user(is_enabled=False)
        user_repository.get_by_id = AsyncMock(
            side_effect=lambda user_id: (
                _user(user_id=ADMIN_ID, is_admin=True)
                if user_id == ADMIN_ID
                else _user(user_id=USER_ID, is_admin=False)
            )
        )
        user_repository.set_enabled = AsyncMock(return_value=disabled)

        token = create_access_token(ADMIN_ID)
        response = client.post(
            f"/admin/users/{USER_ID}/disable",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["is_enabled"] is False
        user_repository.set_enabled.assert_awaited_once_with(USER_ID, enabled=False)
