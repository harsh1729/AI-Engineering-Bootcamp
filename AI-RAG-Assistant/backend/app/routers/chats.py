import uuid

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.deps import get_optional_current_user
from app.database.deps import get_chat_repository, get_message_repository
from app.database.models.chat_session import Chat
from app.database.models.message import Message
from app.database.models.user import User
from app.database.repositories.chat_repository import ChatRepository
from app.database.repositories.message_repository import MessageRepository
from app.models.chat import (
    ChatCreateRequest,
    ChatListResponse,
    ChatMessagesResponse,
    ChatSummary,
    PersistedMessage,
)

router = APIRouter()

_DEFAULT_CHAT_TITLE = "New chat"


def _require_guest_id(guest_id: str | None) -> str:
    if not guest_id:
        raise HTTPException(
            status_code=400,
            detail="guest_id is required for unauthenticated requests.",
        )
    return guest_id


def _to_chat_summary(chat: Chat) -> ChatSummary:
    return ChatSummary(
        id=chat.id,
        guest_id=chat.guest_id,
        user_id=chat.user_id,
        provider=chat.provider,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


def _to_persisted_message(message: Message) -> PersistedMessage:
    return PersistedMessage(
        id=message.id,
        chat_id=message.chat_id,
        role=message.role,
        content=message.content,
        provider=message.provider,
        model=message.model,
        created_at=message.created_at,
    )


@router.post("/chats", response_model=ChatSummary)
async def create_chat(
    request: ChatCreateRequest,
    chat_repository: ChatRepository = Depends(get_chat_repository),
    current_user: User | None = Depends(get_optional_current_user),
) -> ChatSummary:
    """Explicit chat creation (optional). Prefer creating on first message instead."""
    if current_user is not None:
        chat = await chat_repository.create(
            user_id=current_user.id,
            provider=request.provider,
            title=request.title or _DEFAULT_CHAT_TITLE,
        )
    else:
        chat = await chat_repository.create(
            guest_id=_require_guest_id(request.guest_id),
            provider=request.provider,
            title=request.title or _DEFAULT_CHAT_TITLE,
        )
    return _to_chat_summary(chat)


@router.get("/chats", response_model=ChatListResponse)
async def list_chats(
    provider: str = Query(min_length=1),
    guest_id: str | None = Query(default=None, min_length=1),
    chat_repository: ChatRepository = Depends(get_chat_repository),
    current_user: User | None = Depends(get_optional_current_user),
) -> ChatListResponse:
    if current_user is not None:
        if guest_id:
            chats = await chat_repository.list_by_user_and_guest_and_provider(
                current_user.id,
                guest_id,
                provider,
            )
        else:
            chats = await chat_repository.list_by_user_and_provider(
                current_user.id,
                provider,
            )
    else:
        chats = await chat_repository.list_by_guest_and_provider(
            _require_guest_id(guest_id),
            provider,
        )
    return ChatListResponse(chats=[_to_chat_summary(chat) for chat in chats])


@router.get("/chats/{chat_id}/messages", response_model=ChatMessagesResponse)
async def list_chat_messages(
    chat_id: uuid.UUID,
    provider: str = Query(min_length=1),
    guest_id: str | None = Query(default=None, min_length=1),
    chat_repository: ChatRepository = Depends(get_chat_repository),
    message_repository: MessageRepository = Depends(get_message_repository),
    current_user: User | None = Depends(get_optional_current_user),
) -> ChatMessagesResponse:
    if current_user is not None:
        if guest_id:
            chat = await chat_repository.get_by_id_for_user_or_guest(
                chat_id,
                current_user.id,
                guest_id,
                provider=provider,
            )
        else:
            chat = await chat_repository.get_by_id_for_user(
                chat_id,
                current_user.id,
                provider=provider,
            )
    else:
        chat = await chat_repository.get_by_id_for_guest(
            chat_id,
            _require_guest_id(guest_id),
            provider=provider,
        )

    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' was not found.")

    messages = await message_repository.list_by_chat(chat_id)
    return ChatMessagesResponse(
        chat_id=chat_id,
        messages=[_to_persisted_message(message) for message in messages],
    )
