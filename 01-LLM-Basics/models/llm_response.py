from typing import Any

from pydantic import BaseModel, Field
from enums import MessageRole
from .llm_usage import LLMUsage
from .tools.llm_tool_call import LLMToolCall

class LLMResponse(BaseModel):
    id: str
    provider: str
    model: str

    # when it is tools call, we don't get role and text. That is why it is optional
    role: MessageRole | None = None 

    text: str | None = None

    finish_reason: str | None = None

    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    reasoning: str | None = None

    raw_response: Any | None = None

    usage: LLMUsage | None = None