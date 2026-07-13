from google.genai import types

from config import GEMINI_MODEL
from enums import MessageRole, ProviderType
from models import LLMResponse, LLMUsage
from serializers import BaseResponseSerializer


class GeminiResponseSerializer(BaseResponseSerializer):

    def serialize(
        self,
        response: types.GenerateContentResponse,
    ) -> LLMResponse:

        usage = None
        if response.usage_metadata:
            usage = LLMUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

        text = response.text if response.text else None

        return LLMResponse(
            id=response.response_id,
            provider=ProviderType.GEMINI,
            model=response.model_version or GEMINI_MODEL,
            role=MessageRole.ASSISTANT if text else None,
            text=text,
            finish_reason=finish_reason,
            tool_calls=[],
            reasoning=None,
            raw_response=response,
            usage=usage,
        )
