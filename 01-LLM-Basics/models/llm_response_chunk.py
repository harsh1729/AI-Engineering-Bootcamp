from pydantic import BaseModel


class LLMResponseChunk(BaseModel):
    text: str
    is_finished: bool = False