import json
from openai.types.responses import ResponseFunctionToolCall

from serializers import BaseResponseSerializer
from models import LLMResponse,LLMUsage
from models.tools import LLMToolCall
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

        output = response.output[0]

        return LLMResponse(
            id=response.id,
            provider=ProviderType.OPENAI,
            model=response.model,

            #we may not receive text or role like in case of tools call
            role=getattr(output, "role", None),
            text = response.output_text if response.output_text else None,

            finish_reason=response.status, 
            tool_calls=[
                LLMToolCall(
                    id=output.id,
                    call_id=output.call_id,
                    name=output.name,
                    arguments=json.loads(output.arguments),
                )
            ] if isinstance(output, ResponseFunctionToolCall) else [],
            reasoning=None,
            raw_response=response,
            usage=usage,
        )