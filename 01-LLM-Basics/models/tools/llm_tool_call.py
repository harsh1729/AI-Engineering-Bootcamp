from typing import Any

from pydantic import BaseModel


class LLMToolCall(BaseModel):
    id: str
    call_id: str
    name: str
    arguments: dict[str, Any]