import uuid

from sqlalchemy import or_, select

from app.database.models.chat_session import Chat
from app.database.repositories.base import BaseRepository


class ChatRepository(BaseRepository):
    async def create(
        self,
        *,
        title: str,
        user_id: uuid.UUID | None = None,
        guest_id: str | None = None,
        provider: str | None = None,
    ) -> Chat:
        if user_id is None and guest_id is None:
            raise ValueError("Either user_id or guest_id must be provided.")

        chat = Chat(
            user_id=user_id,
            guest_id=guest_id,
            provider=provider,
            title=title,
        )
        self._session.add(chat)
        await self._session.flush()
        return chat

    async def get_by_id(self, chat_id: uuid.UUID) -> Chat | None:
        return await self._session.get(Chat, chat_id)

    async def get_by_id_for_guest(
        self,
        chat_id: uuid.UUID,
        guest_id: str,
        *,
        provider: str | None = None,
    ) -> Chat | None:
        chat = await self.get_by_id(chat_id)
        if chat is None or chat.guest_id != guest_id:
            return None
        if provider is not None and chat.provider != provider:
            return None
        return chat

    async def get_by_id_for_user(
        self,
        chat_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        provider: str | None = None,
    ) -> Chat | None:
        chat = await self.get_by_id(chat_id)
        if chat is None or chat.user_id != user_id:
            return None
        if provider is not None and chat.provider != provider:
            return None
        return chat

    async def get_by_id_for_user_or_guest(
        self,
        chat_id: uuid.UUID,
        user_id: uuid.UUID,
        guest_id: str,
        *,
        provider: str | None = None,
    ) -> Chat | None:
        chat = await self.get_by_id(chat_id)
        if chat is None:
            return None
        if chat.user_id != user_id and chat.guest_id != guest_id:
            return None
        if provider is not None and chat.provider != provider:
            return None
        return chat

    async def touch(self, chat: Chat) -> None:
        await self._session.flush()

    async def list_by_user(self, user_id: uuid.UUID) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user_and_provider(
        self,
        user_id: uuid.UUID,
        provider: str,
    ) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(Chat.user_id == user_id, Chat.provider == provider)
            .order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_guest(self, guest_id: str) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(Chat.guest_id == guest_id)
            .order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_guest_and_provider(
        self,
        guest_id: str,
        provider: str,
    ) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(Chat.guest_id == guest_id, Chat.provider == provider)
            .order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user_and_guest_and_provider(
        self,
        user_id: uuid.UUID,
        guest_id: str,
        provider: str,
    ) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(
                Chat.provider == provider,
                or_(Chat.user_id == user_id, Chat.guest_id == guest_id),
            )
            .order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
