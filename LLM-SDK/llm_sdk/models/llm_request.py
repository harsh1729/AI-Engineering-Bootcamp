from pydantic import BaseModel, Field

from llm_sdk.enums import ProviderType
from llm_sdk.models import LLMMessage
from llm_sdk.models.tools import LLMTool

class LLMRequest(BaseModel):
    messages: list[LLMMessage]

    temperature: float = 1.0
    max_tokens: int = 1024

    tools: list[LLMTool] = Field(default_factory=list)

    # Optional per-request provider override. When omitted, ProviderFactory
    # falls back to the configured default provider (LLM_PROVIDER).
    provider: ProviderType | None = None

    # Optional per-request model override. When omitted, providers fall back
    # to their configured default model.
    model: str | None = None