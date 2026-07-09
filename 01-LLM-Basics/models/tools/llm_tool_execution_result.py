from typing import Any

from pydantic import BaseModel

from .llm_tool_call import LLMToolCall


class LLMToolExecutionResult(BaseModel):
    tool_call: LLMToolCall
    result: Any | None = None
    error: str | None = None