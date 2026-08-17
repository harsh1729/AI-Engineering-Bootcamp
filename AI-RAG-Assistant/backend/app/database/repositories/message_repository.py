import uuid

from sqlalchemy import select

from app.database.models.message import Message
from app.database.repositories.base import BaseRepository


class MessageRepository(BaseRepository):
    async def create(
        self,
        *,
        chat_id: uuid.UUID,
        role: str,
        content: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> Message:
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            provider=provider,
            model=model,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def get_by_id(self, message_id: uuid.UUID) -> Message | None:
        return await self._session.get(Message, message_id)

    async def list_by_chat(self, chat_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
