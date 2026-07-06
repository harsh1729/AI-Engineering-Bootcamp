from typing import Any

from pydantic import BaseModel, Field
from enums import MessageRole
from .llm_usage import LLMUsage

class LLMResponse(BaseModel):
    id: str
    provider: str
    model: str
    role: MessageRole

    text: str

    finish_reason: str | None = None

    tool_calls: list = Field(default_factory=list)

    reasoning: str | None = None

    raw_response: Any | None = None

    usage: LLMUsage | None = None