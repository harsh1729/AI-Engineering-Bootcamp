from google.genai import types

from llm_sdk.config import GEMINI_MODEL
from llm_sdk.enums import MessageRole, ProviderType
from llm_sdk.models import LLMResponse, LLMUsage
from llm_sdk.models.tools import LLMToolCall
from llm_sdk.serializers import BaseResponseSerializer


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
        tool_calls: list[LLMToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = str(candidate.finish_reason)

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if not part.function_call:
                        continue

                    if part.function_call.id is None:
                        raise ValueError(
                            f"Gemini function_call for '{part.function_call.name}' is missing id."
                        )

                    tool_calls.append(
                        LLMToolCall(
                            call_id=part.function_call.id,
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args),
                        )
                    )

        text = self._extract_text(response)

        return LLMResponse(
            id=response.response_id,
            provider=ProviderType.GEMINI,
            model=response.model_version or GEMINI_MODEL,
            role=MessageRole.ASSISTANT if text or tool_calls else None,
            text=text,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning=None,
            raw_response=response,
            usage=usage,
        )

    def _extract_text(
        self,
        response: types.GenerateContentResponse,
    ) -> str | None:

        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            return None

        text_parts = [
            part.text
            for part in response.candidates[0].content.parts
            if part.text
        ]

        return "".join(text_parts) if text_parts else None
