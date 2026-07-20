import json
from openai.types.responses import ResponseFunctionToolCall

from llm_sdk.serializers import BaseResponseSerializer
from llm_sdk.models import LLMResponse,LLMUsage
from llm_sdk.models.tools import LLMToolCall
from llm_sdk.enums import ProviderType


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

        # response.status is "incomplete" for both truncated-by-length and
        # content-filter cases. incomplete_details.reason tells us which,
        # e.g. "max_output_tokens" - mirroring how Claude reports "max_tokens"
        # directly as its stop_reason.
        finish_reason = response.status
        if response.status == "incomplete" and response.incomplete_details:
            finish_reason = response.incomplete_details.reason or finish_reason

        return LLMResponse(
            id=response.id,
            provider=ProviderType.OPENAI,
            model=response.model,

            role=assistant_message.role if assistant_message else None,
            text=response.output_text if response.output_text else None,

            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning=None,
            raw_response=response,
            usage=usage,
        )