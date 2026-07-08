import json
from openai.types.responses import ResponseFunctionToolCall

from serializers import BaseResponseSerializer
from models import LLMResponse,LLMUsage
from models.tools import LLMToolCall
from enums import ProviderType


class OpenAIResponseSerializer(BaseResponseSerializer):

    def serialize(
    self,
    response,
) -> LLMResponse:

        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        tool_calls = [
            LLMToolCall(
                id=item.id,
                call_id=item.call_id,
                name=item.name,
                arguments=json.loads(item.arguments),
            )
            for item in response.output
            if isinstance(item, ResponseFunctionToolCall)
        ]

        assistant_message = next(
            (item for item in response.output if getattr(item, "role", None)),
            None,
        )

        return LLMResponse(
            id=response.id,
            provider=ProviderType.OPENAI,
            model=response.model,

            role=assistant_message.role if assistant_message else None,
            text=response.output_text if response.output_text else None,

            finish_reason=response.status,
            tool_calls=tool_calls,
            reasoning=None,
            raw_response=response,
            usage=usage,
        )