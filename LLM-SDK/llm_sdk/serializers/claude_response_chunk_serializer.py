from anthropic.types import RawContentBlockDeltaEvent,RawMessageDeltaEvent,TextDelta,InputJSONDelta

from llm_sdk.models import LLMResponseChunk
from llm_sdk.serializers import BaseResponseChunkSerializer


class ClaudeResponseChunkSerializer(BaseResponseChunkSerializer):


    
    def serialize(
        self,
        event,
    ) -> LLMResponseChunk | None:

        #
        # Stream assistant text.
        #
        if isinstance(event, RawContentBlockDeltaEvent):

            if isinstance(event.delta, TextDelta):
                return LLMResponseChunk(
                    text=event.delta.text,
                )

            #
            # Tool arguments are reconstructed by ClaudeProvider.
            #
            if isinstance(event.delta, InputJSONDelta):
                return None

        #
        # Mark stream finished only when Claude finishes its turn.
        #
        if isinstance(event, RawMessageDeltaEvent):
            if event.delta.stop_reason == "end_turn":
                return LLMResponseChunk(
                    is_finished=True,
                )

        return None