from pydantic import BaseModel

from .tools.llm_tool_call import LLMToolCall

class LLMResponseChunk(BaseModel):
    text: str | None = None
    
    tool_call: LLMToolCall | None = None

    is_finished: bool = False