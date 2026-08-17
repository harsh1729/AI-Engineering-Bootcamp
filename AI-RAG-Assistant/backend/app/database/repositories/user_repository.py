import uuid

from sqlalchemy import select

from app.database.models._types import utcnow
from app.database.models.user import User
from app.database.repositories.base import BaseRepository


class UserRepository(BaseRepository):
    async def create(
        self,
        *,
        name: str,
        email: str,
        password_hash: str,
        is_approved: bool = False,
        is_admin: bool = False,
        is_enabled: bool = True,
    ) -> User:
        user = User(
            name=name,
            email=email,
            password_hash=password_hash,
            is_approved=is_approved,
            is_admin=is_admin,
            is_enabled=is_enabled,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self._session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_pending_non_admin(self) -> list[User]:
        stmt = (
            select(User)
            .where(User.is_approved.is_(False), User.is_admin.is_(False))
            .order_by(User.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_approved_non_admin(self) -> list[User]:
        stmt = (
            select(User)
            .where(User.is_approved.is_(True), User.is_admin.is_(False))
            .order_by(User.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def approve(self, user_id: uuid.UUID) -> User | None:
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        user.is_approved = True
        user.is_enabled = True
        user.updated_at = utcnow()
        await self._session.flush()
        return user

    async def set_enabled(self, user_id: uuid.UUID, *, enabled: bool) -> User | None:
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        user.is_enabled = enabled
        user.updated_at = utcnow()
        await self._session.flush()
        return user

    async def update_last_login(self, user_id: uuid.UUID) -> User | None:
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        user.last_login_at = utcnow()
        await self._session.flush()
        return user
