import uuid

from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.models.auth import AdminUserSummary, AdminUsersResponse


class UserNotFoundError(Exception):
    """Raised when an admin action targets a user that does not exist."""


class UserNotManageableError(Exception):
    """Raised when an admin action targets an admin account or themselves."""


class AdminUserService:
    def __init__(self, user_repository: UserRepository) -> None:
        self._user_repository = user_repository

    async def list_users(self) -> AdminUsersResponse:
        pending = await self._user_repository.list_pending_non_admin()
        approved = await self._user_repository.list_approved_non_admin()
        return AdminUsersResponse(
            pending=[self._to_summary(user) for user in pending],
            approved=[self._to_summary(user) for user in approved],
        )

    async def approve_user(
        self,
        user_id: uuid.UUID,
        *,
        acting_admin_id: uuid.UUID,
    ) -> AdminUserSummary:
        user = await self._get_manageable_user(user_id, acting_admin_id=acting_admin_id)
        approved = await self._user_repository.approve(user.id)
        if approved is None:
            raise UserNotFoundError(f"User '{user_id}' was not found.")
        return self._to_summary(approved)

    async def disable_user(
        self,
        user_id: uuid.UUID,
        *,
        acting_admin_id: uuid.UUID,
    ) -> AdminUserSummary:
        user = await self._get_manageable_user(user_id, acting_admin_id=acting_admin_id)
        updated = await self._user_repository.set_enabled(user.id, enabled=False)
        if updated is None:
            raise UserNotFoundError(f"User '{user_id}' was not found.")
        return self._to_summary(updated)

    async def enable_user(
        self,
        user_id: uuid.UUID,
        *,
        acting_admin_id: uuid.UUID,
    ) -> AdminUserSummary:
        user = await self._get_manageable_user(user_id, acting_admin_id=acting_admin_id)
        if not user.is_approved:
            raise UserNotManageableError("Only approved users can be re-enabled.")

        updated = await self._user_repository.set_enabled(user.id, enabled=True)
        if updated is None:
            raise UserNotFoundError(f"User '{user_id}' was not found.")
        return self._to_summary(updated)

    async def _get_manageable_user(
        self,
        user_id: uuid.UUID,
        *,
        acting_admin_id: uuid.UUID,
    ) -> User:
        if user_id == acting_admin_id:
            raise UserNotManageableError("You cannot manage your own account here.")

        user = await self._user_repository.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"User '{user_id}' was not found.")

        if user.is_admin:
            raise UserNotManageableError("Admin accounts cannot be managed here.")

        return user

    @staticmethod
    def _to_summary(user: User) -> AdminUserSummary:
        return AdminUserSummary(
            id=user.id,
            name=user.name,
            email=user.email,
            is_enabled=user.is_enabled,
            created_at=user.created_at,
        )
