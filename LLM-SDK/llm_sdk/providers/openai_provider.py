from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError
from typing import Generator

from openai.types.responses import ResponseCreatedEvent

from llm_sdk.config import OPENAI_API_KEY
from llm_sdk.providers import LLMProvider
from llm_sdk.models import LLMResponse,LLMRequest,LLMResponseChunk
from llm_sdk.serializers import OpenAIRequestSerializer,OpenAIResponseSerializer,OpenAIResponseChunkSerializer
from llm_sdk.models.tools import LLMToolCall

from llm_sdk.logs import get_logger
logger = get_logger(__name__)

class OpenAIProvider(LLMProvider):

    

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.request_serializer = OpenAIRequestSerializer()
        self.response_serializer = OpenAIResponseSerializer()
        self.response_chunk_serializer = OpenAIResponseChunkSerializer()

    # Generate a complete response for the given prompt.
    def generate_response(self, request: LLMRequest) -> LLMResponse:

        payload = self.request_serializer.serialize(request)

        client_response = self._call_with_retry(
            lambda _: self.client.responses.create(**payload),
            request=request,
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
                    request=request,
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
            request=request,
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
        tools_enabled = bool(request.tools)
        tools_executed = False

        for _ in range(MAX_TOOL_CALL_DEPTH):

            stream_response_id = None
            tool_calls: list[LLMToolCall] = []
            buffered_text: list[str] = []

            for event in stream:

                if isinstance(event, ResponseCreatedEvent):
                    stream_response_id = event.response.id
                    continue

                chunk = self.response_chunk_serializer.serialize(event)

                if not chunk:
                    continue

                if chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    continue

                if chunk.text:
                    if tools_enabled and not tools_executed:
                        buffered_text.append(chunk.text)
                    else:
                        yield chunk

            if len(tool_calls) > 0:
                tools_executed = True
                tool_results = self._execute_tools(tool_calls)

                payload = self.request_serializer.serialize_tool_results(
                    request=request,
                    previous_response_id=stream_response_id,
                    tool_results=tool_results,
                )

                stream = self._call_with_retry(
                    lambda _: self.client.responses.create(**payload, stream=True),
                    request=request,
                    operation="OpenAI streaming follow-up request",
                )

                continue

            if buffered_text:
                for text in buffered_text:
                    yield LLMResponseChunk(text=text)

            return

        raise RuntimeError(
            f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
        )

    def _is_retryable_error(self, exc: Exception) -> bool:

        if isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError)):
            return True

        return isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503)