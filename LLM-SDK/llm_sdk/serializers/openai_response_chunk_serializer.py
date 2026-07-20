from openai.types.responses import ResponseTextDeltaEvent,ResponseOutputItemDoneEvent,ResponseFunctionToolCall
import json

from llm_sdk.models import LLMResponseChunk
from llm_sdk.serializers import BaseResponseChunkSerializer
from llm_sdk.models.tools import LLMToolCall


class OpenAIResponseChunkSerializer(BaseResponseChunkSerializer):

    def serialize(
        self,
        event,
    ) -> LLMResponseChunk | None:
        if isinstance(event, ResponseTextDeltaEvent):

            return LLMResponseChunk(
                text=event.delta,
            )
        
        #if it's a tool call we read ResponseOutputItemDoneEvent & ResponseFunctionToolCall rather than ResponseTextDeltaEvent
        if (
            isinstance(event, ResponseOutputItemDoneEvent)
            and isinstance(event.item, ResponseFunctionToolCall)
        ):

            return LLMResponseChunk(
                tool_call=
                    LLMToolCall(
                        id=event.item.id,
                        call_id=event.item.call_id,
                        name=event.item.name,
                        arguments=json.loads(event.item.arguments),
                    )
                
            )

        return None