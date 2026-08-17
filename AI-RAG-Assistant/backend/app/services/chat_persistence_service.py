import uuid

from llm_sdk.enums import MessageRole

from app.database.models.chat_session import Chat
from app.database.models._types import utcnow
from app.database.repositories.chat_repository import ChatRepository
from app.database.repositories.message_repository import MessageRepository
from app.models.chat import ChatRequest
from app.services.chat_persistence_exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
)

_MAX_CHAT_TITLE_LENGTH = 255


def extract_latest_user_message(request: ChatRequest) -> str:
    for message in reversed(request.messages):
        if message.role == MessageRole.USER:
            return message.content
    raise ValueError("At least one user message is required.")


def title_from_message(content: str) -> str:
    normalized = " ".join(content.split())
    if not normalized:
        raise ValueError("Message content is required to create a chat.")
    if len(normalized) <= _MAX_CHAT_TITLE_LENGTH:
        return normalized
    return normalized[: _MAX_CHAT_TITLE_LENGTH - 1] + "…"


class ChatPersistenceService:
    """Persists guest- or user-scoped chats and messages to PostgreSQL."""

    def __init__(
        self,
        chat_repository: ChatRepository,
        message_repository: MessageRepository,
    ) -> None:
        self._chat_repository = chat_repository
        self._message_repository = message_repository

    async def get_or_create_chat(
        self,
        *,
        provider: str,
        guest_id: str | None = None,
        user_id: uuid.UUID | None = None,
        chat_id: uuid.UUID | None = None,
        first_message: str | None = None,
    ) -> Chat:
        if user_id is None and guest_id is None:
            raise ValueError("Either user_id or guest_id must be provided.")

        if chat_id is not None:
            chat = await self._chat_repository.get_by_id(chat_id)
            if chat is None:
                raise ChatNotFoundError(f"Chat '{chat_id}' was not found.")
            if not self._can_access_chat(chat, user_id=user_id, guest_id=guest_id):
                owner = "user" if user_id is not None else "guest"
                raise ChatAccessDeniedError(
                    f"Chat '{chat_id}' does not belong to this {owner}."
                )
            if chat.provider != provider:
                raise ChatAccessDeniedError(
                    f"Chat '{chat_id}' does not belong to provider '{provider}'."
                )
            return chat

        if not first_message:
            raise ValueError("first_message is required when creating a new chat.")

        return await self._chat_repository.create(
            user_id=user_id,
            guest_id=None if user_id is not None else guest_id,
            provider=provider,
            title=title_from_message(first_message),
        )

    @staticmethod
    def _can_access_chat(
        chat: Chat,
        *,
        user_id: uuid.UUID | None,
        guest_id: str | None,
    ) -> bool:
        if user_id is not None:
            if chat.user_id == user_id:
                return True
            return guest_id is not None and chat.guest_id == guest_id

        return guest_id is not None and chat.guest_id == guest_id

    async def append_user_message(self, chat: Chat, content: str) -> None:
        await self._message_repository.create(
            chat_id=chat.id,
            role="user",
            content=content,
        )
        await self._touch_chat(chat)

    async def append_assistant_message(
        self,
        chat: Chat,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        await self._message_repository.create(
            chat_id=chat.id,
            role="assistant",
            content=content,
            provider=provider,
            model=model,
        )
        await self._touch_chat(chat)

    async def _touch_chat(self, chat: Chat) -> None:
        chat.updated_at = utcnow()
        await self._chat_repository.touch(chat)
