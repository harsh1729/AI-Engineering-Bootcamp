from google.genai.types import FinishReason, GenerateContentResponse, Part

from models import LLMResponseChunk
from serializers import BaseResponseChunkSerializer


class GeminiResponseChunkSerializer(BaseResponseChunkSerializer):

    def serialize(
        self,
        chunk: GenerateContentResponse,
    ) -> LLMResponseChunk | None:

        text = self._extract_text(chunk)

        if text:
            return LLMResponseChunk(
                text=text,
            )

        if chunk.candidates:
            candidate = chunk.candidates[0]
            finish_reason = candidate.finish_reason
            has_function_call = self._has_function_call(candidate.content.parts if candidate.content else [])

            if finish_reason in (FinishReason.STOP, FinishReason.MAX_TOKENS) and not has_function_call:
                return LLMResponseChunk(
                    is_finished=True,
                )

        return None

    def _extract_text(
        self,
        chunk: GenerateContentResponse,
    ) -> str | None:

        if (
            not chunk.candidates
            or not chunk.candidates[0].content
            or not chunk.candidates[0].content.parts
        ):
            return None

        text_parts = [
            part.text
            for part in chunk.candidates[0].content.parts
            if part.text
        ]

        return "".join(text_parts) if text_parts else None

    def _has_function_call(
        self,
        parts: list[Part],
    ) -> bool:

        return any(part.function_call for part in parts)
