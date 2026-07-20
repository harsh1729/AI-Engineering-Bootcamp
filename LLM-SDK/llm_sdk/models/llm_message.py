from pydantic import BaseModel,Field

from llm_sdk.enums import MessageRole
from llm_sdk.models.tools import LLMToolCall


class LLMMessage(BaseModel):
    role: MessageRole
    content: str


    # Present only on assistant messages that request tool execution
    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    # Present only on tool messages
    tool_call_id: str | None = None