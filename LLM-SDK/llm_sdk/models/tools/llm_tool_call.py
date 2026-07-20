from typing import Any

from pydantic import BaseModel


class LLMToolCall(BaseModel):
    id: str | None = None #Present for OpenAI but not for Claude
    call_id: str #ID retuned back to LLM model for toolcall identification
    name: str
    arguments: dict[str, Any]