from anthropic.types import TextBlock, ToolUseBlock

from serializers import BaseResponseSerializer
from models import LLMResponse, LLMUsage
from models.tools import LLMToolCall
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

        tool_calls = [
            LLMToolCall(
                id=block.id,
                call_id=block.id,
                name=block.name,
                arguments=block.input,
            )
            for block in response.content
            if isinstance(block, ToolUseBlock)
        ]

        text_blocks = [
            block.text
            for block in response.content
            if isinstance(block, TextBlock)
        ]

        return LLMResponse(
            id=response.id,
            provider=ProviderType.CLAUDE,
            model=response.model,
            role=response.role,
            text="\n".join(text_blocks) if text_blocks else None,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls,
            reasoning=None,
            raw_response=response,
            usage=usage,
        )

