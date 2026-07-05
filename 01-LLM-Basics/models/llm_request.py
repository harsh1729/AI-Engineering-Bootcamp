from pydantic import BaseModel, Field

from models import LLMMessage


class LLMRequest(BaseModel):
    messages: list[LLMMessage]

    temperature: float = 1.0
    max_tokens: int = 1024

    tools: list = Field(default_factory=list)