from serializers import BaseResponseSerializer
from models import LLMResponse,LLMUsage
from enums import ProviderType

class ClaudeResponseSerializer(BaseResponseSerializer):

    def serialize(
        self,
        response,
    ) -> LLMResponse:
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            id=response.id,
            provider=ProviderType.CLAUDE,
            model=response.model,
            role=response.role,
            text=response.content[0].text,
            finish_reason=response.stop_reason, 
            tool_calls=[],
            reasoning=None,
            raw_response=response,
            usage=usage,
        )
