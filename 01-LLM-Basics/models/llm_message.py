from pydantic import BaseModel,Field

from enums import MessageRole
from models.tools import LLMToolCall


class LLMMessage(BaseModel):
    role: MessageRole
    content: str


    # Present only on assistant messages that request tool execution
    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    # Present only on tool messages
    tool_call_id: str | None = None