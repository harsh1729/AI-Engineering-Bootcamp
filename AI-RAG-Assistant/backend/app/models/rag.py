from pydantic import BaseModel, Field, field_validator

from llm_sdk.enums import ProviderType

from app.context.context_models import ContextSource


def _normalize_enum_value(enum_cls, value: object) -> object:
    if value is None or isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        try:
            return enum_cls[value.upper()]
        except KeyError:
            pass

    return value


class RAGRequest(BaseModel):
    """Input for a grounded question-answering request."""

    query: str
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, gt=0)
    provider: ProviderType | None = None
    model: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return _normalize_enum_value(ProviderType, value)


class RAGResponse(BaseModel):
    """Grounded answer plus source metadata from retrieval."""

    answer: str
    sources: list[ContextSource]
    warning: str | None = None
