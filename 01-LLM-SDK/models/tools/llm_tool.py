from pydantic import BaseModel, Field

from .llm_tool_params import LLMToolParam


class LLMTool(BaseModel):
    name: str
    description: str

    parameters: list[LLMToolParam] = Field(default_factory=list)