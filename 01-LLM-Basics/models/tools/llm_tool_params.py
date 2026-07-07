from pydantic import BaseModel


class LLMToolParam(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True