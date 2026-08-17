from app.config import (
    ACCOUNT_DISABLED_MESSAGE,
    ACCOUNT_PENDING_APPROVAL_MESSAGE,
    REGISTRATION_PENDING_MESSAGE,
)
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.models.auth import AuthResponse, LoginRequest, RegisterRequest, UserPublic
from app.services.jwt_service import create_access_token
from app.services.password_service import hash_password, verify_password


class EmailAlreadyRegisteredError(Exception):
    """Raised when register is called with an email that already exists."""


class InvalidCredentialsError(Exception):
    """Raised when login email/password do not match a user."""


class AccountNotApprovedError(Exception):
    """Raised when a user exists but is not approved to sign in."""


class AccountDisabledError(Exception):
    """Raised when a user account has been disabled by an admin."""


class AuthService:
    def __init__(self, user_repository: UserRepository) -> None:
        self._user_repository = user_repository

    async def register(self, request: RegisterRequest) -> str:
        existing = await self._user_repository.get_by_email(request.email.lower())
        if existing is not None:
            raise EmailAlreadyRegisteredError(
                "An account with this email already exists."
            )

        await self._user_repository.create(
            name=request.name.strip(),
            email=request.email.lower(),
            password_hash=hash_password(request.password),
            is_approved=False,
            is_enabled=True,
        )
        return REGISTRATION_PENDING_MESSAGE

    async def login(self, request: LoginRequest) -> AuthResponse:
        user = await self._user_repository.get_by_email(request.email.lower())
        if user is None or not verify_password(request.password, user.password_hash):
            raise InvalidCredentialsError("Invalid email or password.")

        if not user.is_approved:
            raise AccountNotApprovedError(ACCOUNT_PENDING_APPROVAL_MESSAGE)

        if not user.is_enabled:
            raise AccountDisabledError(ACCOUNT_DISABLED_MESSAGE)

        await self._user_repository.update_last_login(user.id)
        return self._build_auth_response(user)

    def _build_auth_response(self, user: User) -> AuthResponse:
        return AuthResponse(
            access_token=create_access_token(user.id),
            user=UserPublic(
                id=user.id,
                name=user.name,
                email=user.email,
                is_admin=user.is_admin,
            ),
        )
