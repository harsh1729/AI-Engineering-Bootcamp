from anthropic import Anthropic

from config import ANTHROPIC_API_KEY,ANTHROPIC_MODEL
from providers import LLMProvider
from models import LLMResponse,LLMRequest
from enums import ProviderType
from serializers import ClaudeMessageSerializer


class ClaudeProvider(LLMProvider):

    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.serializer = ClaudeMessageSerializer()

    def generate(self, request: LLMRequest) -> LLMResponse:

        payload = self.serializer.serialize_messages(
            request.messages
        )

        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=request.max_tokens,
            system=payload["system"],
            messages=payload["messages"],
        )

        return LLMResponse(
            id=response.id,
            provider=ProviderType.CLAUDE,
            model=response.model,
            role=response.role,
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
            tool_calls=[],
            reasoning=None,
            raw_response=response,
        )