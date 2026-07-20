from pydantic import BaseModel, field_validator

from llm_sdk.enums import MessageRole, ProviderType


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
    guest_id: str
    messages: list[ChatMessage]
    provider: ProviderType | None = None
    model: str | None = None
    # IDs of documents (from POST /documents/upload) the user wants this
    # message to draw on. Not used yet - retrieval/parsing come later. Kept
    # out of LLMRequest so the SDK never needs to know documents exist.
    document_ids: list[str] = []

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return _normalize_enum_value(ProviderType, value)


class ChatResponse(BaseModel):
    response: str
    warning: str | None = None
