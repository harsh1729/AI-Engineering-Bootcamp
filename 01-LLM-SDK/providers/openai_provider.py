from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError
from typing import Generator

from openai.types.responses import ResponseCreatedEvent

from config import OPENAI_API_KEY
from providers import LLMProvider
from models import LLMResponse,LLMRequest,LLMResponseChunk
from serializers import OpenAIRequestSerializer,OpenAIResponseSerializer,OpenAIResponseChunkSerializer
from models.tools import LLMToolCall

from logs import get_logger
logger = get_logger(__name__)

class OpenAIProvider(LLMProvider):

    

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.request_serializer = OpenAIRequestSerializer()
        self.response_serializer = OpenAIResponseSerializer()
        self.response_chunk_serializer = OpenAIResponseChunkSerializer()

    # Generate a complete response for the given prompt.
    def generate_response(self, request: LLMRequest) -> LLMResponse:

        payload = self.request_serializer.serialize(request)

        client_response = self._call_with_retry(
            lambda _: self.client.responses.create(**payload),
            operation="OpenAI API request",
        )

        #Parse to our data model
        llm_response = self.response_serializer.serialize(client_response)

        return self._finalize_response(request,llm_response)

        
    
    
    def _finalize_response(self,request: LLMRequest, response: LLMResponse)-> LLMResponse:

        while response.tool_calls:


            tool_results = self._execute_tools(response.tool_calls)

            payload = self.request_serializer.serialize_tool_results(
                request=request,
                previous_response_id=response.id,
                tool_results=tool_results,
            )

            response = self.response_serializer.serialize(
                self._call_with_retry(
                    lambda _: self.client.responses.create(**payload),
                    operation="OpenAI API follow-up request",
                )
            )

        return response
    


     # Generate a stream of response for the given prompt.
    def generate_stream(
    self,
    request: LLMRequest,
    )-> Generator[LLMResponseChunk, None, None]:

        payload = self.request_serializer.serialize(request)
         
        payload["stream"] = True
         
        client_stream_response = self._call_with_retry(
            lambda _: self.client.responses.create(**payload),
            operation="OpenAI streaming API request",
        )

        yield from self._finalize_stream(request,client_stream_response)

        # for event in stream:
        #     chunk = self.response_chunk_serializer.serialize(event)
        #     if chunk:
        #         yield chunk

    

    def _finalize_stream(
    self,
    request: LLMRequest,
    stream,
    ) -> Generator[LLMResponseChunk, None, None]:

        MAX_TOOL_CALL_DEPTH = 10

        for _ in range(MAX_TOOL_CALL_DEPTH):

            stream_response_id = None
            tool_calls: list[LLMToolCall] = []

            for event in stream:

                if isinstance(event, ResponseCreatedEvent):
                    stream_response_id = event.response.id
                    continue

                chunk = self.response_chunk_serializer.serialize(event)

                if not chunk:
                    continue

                if chunk.text:
                    yield chunk

                if chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    continue

            if len(tool_calls) > 0:
                tool_results = self._execute_tools(tool_calls)

                payload = self.request_serializer.serialize_tool_results(
                    request=request,
                    previous_response_id=stream_response_id,
                    tool_results=tool_results,
                )

                

                stream = self._call_with_retry(
                    lambda _: self.client.responses.create(**payload, stream=True),
                    operation="OpenAI streaming follow-up request",
                )


                continue

            # No tool calls OR chunk.text means we're done 
            return

        raise RuntimeError(
            f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
        )

    def _is_retryable_error(self, exc: Exception) -> bool:

        if isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError)):
            return True

        return isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503)