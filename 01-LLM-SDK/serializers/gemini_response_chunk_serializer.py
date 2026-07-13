from google.genai.types import FinishReason, GenerateContentResponse

from models import LLMResponseChunk
from serializers import BaseResponseChunkSerializer


class GeminiResponseChunkSerializer(BaseResponseChunkSerializer):

    def serialize(
        self,
        chunk: GenerateContentResponse,
    ) -> LLMResponseChunk | None:

        if chunk.text:
            return LLMResponseChunk(
                text=chunk.text,
            )

        if chunk.candidates:
            finish_reason = chunk.candidates[0].finish_reason
            if finish_reason in (FinishReason.STOP, FinishReason.MAX_TOKENS):
                return LLMResponseChunk(
                    is_finished=True,
                )

        return None
