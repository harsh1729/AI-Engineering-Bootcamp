from pydantic import BaseModel


class LLMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None

        return self.input_tokens + self.output_tokens