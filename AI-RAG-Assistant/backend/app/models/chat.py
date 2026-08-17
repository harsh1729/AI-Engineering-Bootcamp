import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from llm_sdk.enums import MessageRole, ProviderType

from app.context.context_models import ContextSource
from app.models.rag_config import RagOptions


def _normalize_enum_value(enum_cls, value: object) -> object:
    """Accept an enum by name (e.g. "USER") or value (e.g. "user"),
    case-insensitively, so clients don't need to know the exact casing."""
    if value is None or isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        try:
            return enum_cls[value.upper()]
        except KeyError:
            pass

    return value


class ChatMessage(BaseModel):
    role: MessageRole
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: object) -> object:
        return _normalize_enum_value(MessageRole, value)


class ChatRequest(BaseModel):
    guest_id: str | None = None
    messages: list[ChatMessage]
    chat_id: uuid.UUID | None = None
    provider: ProviderType | None = None
    model: str | None = None
    # IDs of documents (from POST /documents/upload) that trigger RAG-backed chat.
    document_ids: list[str] = []
    rag_options: RagOptions | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return _normalize_enum_value(ProviderType, value)


class ChatResponse(BaseModel):
    response: str
    warning: str | None = None
    sources: list[ContextSource] = Field(default_factory=list)
    chat_id: uuid.UUID | None = None


class ChatCreateRequest(BaseModel):
    guest_id: str | None = None
    provider: str
    title: str | None = None


class ChatSummary(BaseModel):
    id: uuid.UUID
    guest_id: str | None = None
    user_id: uuid.UUID | None = None
    provider: str | None = None
    title: str
    created_at: datetime
    updated_at: datetime


class ChatListResponse(BaseModel):
    chats: list[ChatSummary]


class PersistedMessage(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str
    content: str
    provider: str | None = None
    model: str | None = None
    created_at: datetime


class ChatMessagesResponse(BaseModel):
    chat_id: uuid.UUID
    messages: list[PersistedMessage]
