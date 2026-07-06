from serializers import BaseResponseSerializer
from models import LLMResponse,LLMUsage
from enums import ProviderType


class OpenAIResponseSerializer(BaseResponseSerializer):

    def serialize(
        self,
        response ,
    ) -> LLMResponse:
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            id=response.id,
            provider=ProviderType.OPENAI,
            model=response.model,
            role=response.output[0].role,
            text=response.output_text,
            finish_reason=response.status, 
            tool_calls=[],
            reasoning=None,
            raw_response=response,
            usage=usage,
        )