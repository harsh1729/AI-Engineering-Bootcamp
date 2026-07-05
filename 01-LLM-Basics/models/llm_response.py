from typing import Any

from pydantic import BaseModel, Field
from enums import MessageRole

class LLMResponse(BaseModel):
    id: str
    provider: str
    model: str
    role: MessageRole

    text: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    finish_reason: str | None = None

    tool_calls: list = Field(default_factory=list)

    reasoning: str | None = None

    raw_response: Any | None = None