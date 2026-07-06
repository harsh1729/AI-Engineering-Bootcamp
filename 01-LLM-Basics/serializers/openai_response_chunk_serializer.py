from openai.types.responses import ResponseTextDeltaEvent

from models import LLMResponseChunk
from serializers import BaseResponseChunkSerializer


class OpenAIResponseChunkSerializer(BaseResponseChunkSerializer):

    def serialize(
        self,
        event,
    ) -> LLMResponseChunk | None:
        if isinstance(event, ResponseTextDeltaEvent):

            return LLMResponseChunk(
                text=event.delta,
            )

        return None