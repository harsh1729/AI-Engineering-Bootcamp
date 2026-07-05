from pydantic import BaseModel

from enums import MessageRole


class LLMMessage(BaseModel):
    role: MessageRole
    content: str