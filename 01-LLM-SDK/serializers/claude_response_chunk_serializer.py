from anthropic.types import RawContentBlockDeltaEvent

from models import LLMResponseChunk
from serializers import BaseResponseChunkSerializer


class ClaudeResponseChunkSerializer(BaseResponseChunkSerializer):

    def serialize(
        self,
        event,
    ) -> LLMResponseChunk | None:
        if isinstance(event, RawContentBlockDeltaEvent):

            return LLMResponseChunk(
                text=event.delta.text,
            )

        return None