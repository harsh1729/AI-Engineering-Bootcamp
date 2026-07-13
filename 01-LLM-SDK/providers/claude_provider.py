from anthropic import Anthropic, APIConnectionError, APIStatusError, InternalServerError, RateLimitError
from typing import Generator
from pprint import pprint 
from anthropic.types import RawContentBlockStartEvent, RawContentBlockDeltaEvent, RawContentBlockStopEvent,RawMessageDeltaEvent
from anthropic.types import ToolUseBlock,InputJSONDelta
import json

from config import ANTHROPIC_API_KEY
from providers import LLMProvider
from models import LLMResponse,LLMRequest,LLMResponseChunk
from serializers import ClaudeRequestSerializer,ClaudeResponseSerializer,ClaudeResponseChunkSerializer
from models.tools import LLMToolCall

from logs import get_logger
logger = get_logger(__name__)

class ClaudeProvider(LLMProvider):

    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.request_serializer = ClaudeRequestSerializer()
        self.response_serializer = ClaudeResponseSerializer()
        self.response_chunk_serializer = ClaudeResponseChunkSerializer()

    def generate_response(self, request: LLMRequest) -> LLMResponse:

        
        payload = self.request_serializer.serialize(request)

        client_response = self._call_with_retry(
            lambda _: self.client.messages.create(**payload),
            operation="Claude API request",
        )

        llm_response = self.response_serializer.serialize(client_response)
    
        return self._finalize_response(request,llm_response)
    

    def _finalize_response(self,request: LLMRequest, response: LLMResponse)-> LLMResponse:

        while response.tool_calls:


            tool_results = self._execute_tools(response.tool_calls)

            payload = self.request_serializer.serialize_tool_results(
                request=request,
                assistant_content=response.raw_response.content,
                tool_results=tool_results,
            )

            response = self.response_serializer.serialize(
                self._call_with_retry(
                    lambda _: self.client.messages.create(**payload),
                    operation="Claude API follow-up request",
                )
            )

        return response
    
  
    
    def generate_stream(
    self,
    request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:

        payload = self.request_serializer.serialize(request)
        payload["stream"] = True

        client_stream = self._call_with_retry(
            lambda _: self.client.messages.create(**payload),
            operation="Claude streaming API request",
        )

        yield from self._finalize_stream(
            request=request,
            stream=client_stream,
        )

    
    def _finalize_stream(
    self,
    request: LLMRequest,
    stream,
) -> Generator[LLMResponseChunk, None, None]:

        MAX_TOOL_CALL_DEPTH = 10

        for _ in range(MAX_TOOL_CALL_DEPTH):

            tool_calls: list[LLMToolCall] = []

            current_tool_call: LLMToolCall | None = None
            current_tool_arguments_json = ""
            assistant_content: list[dict] = []

            for event in stream:

                #
                # Stream assistant text immediately.
                #
                chunk = self.response_chunk_serializer.serialize(event)

                if chunk and chunk.text :
                    yield chunk

                #
                # Tool block begins.
                #
                if isinstance(event, RawContentBlockStartEvent):

                    content_block = event.content_block

                    if isinstance(content_block, ToolUseBlock):

                        current_tool_call = LLMToolCall(
                            call_id=content_block.id,
                            name=content_block.name,
                            arguments={},
                        )

                        current_tool_arguments_json = ""

                    continue

                #
                # Collect streamed JSON arguments.
                #
                if isinstance(event, RawContentBlockDeltaEvent):

                    delta = event.delta

                    if isinstance(delta, InputJSONDelta):
                        current_tool_arguments_json += delta.partial_json

                    continue

                #
                # Single tool execution complete block.
                #
                if (
                    isinstance(event, RawContentBlockStopEvent)
                    and current_tool_call is not None
                ):
                    
                    try:
                        current_tool_call.arguments = json.loads(
                            current_tool_arguments_json
                        )
                    except json.JSONDecodeError:
                        logger.exception(
                            "Failed to reconstruct streamed Claude tool arguments."
                        )
                        raise

                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": current_tool_call.call_id,
                            "name": current_tool_call.name,
                            "input": current_tool_call.arguments,
                        }
                    )

                    tool_calls.append(current_tool_call)

                    #
                    # Reset state for the next tool block.
                    #
                    current_tool_call = None
                    current_tool_arguments_json = "" 

                    continue

                #
                # Claude finished requesting tool execution.
                #
                if (
                    isinstance(event, RawMessageDeltaEvent)
                    and event.delta.stop_reason == "tool_use"
                ):
                    continue      

            #
            # No tool calls -> streaming is complete.
            #
            if not tool_calls:
                return

            #
            # Execute tools using shared implementation.
            #
            tool_results = self._execute_tools(tool_calls)

            #
            # Build follow-up Claude request.
            #
            payload = self.request_serializer.serialize_tool_results(
                request=request,
                assistant_content=assistant_content,
                tool_results=tool_results,
            )

            #
            # Continue streaming from Claude.
            #
            stream = self._call_with_retry(
                lambda _: self.client.messages.create(**payload, stream=True),
                operation="Claude streaming follow-up request",
            )

        raise RuntimeError(
            f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
        )

    def _is_retryable_error(self, exc: Exception) -> bool:

        if isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError)):
            return True

        return isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503)
