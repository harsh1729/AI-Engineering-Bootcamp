from anthropic import Anthropic
from typing import Generator
from pprint import pprint 

from config import ANTHROPIC_API_KEY
from providers import LLMProvider
from models import LLMResponse,LLMRequest,LLMResponseChunk
from serializers import ClaudeRequestSerializer,ClaudeResponseSerializer,ClaudeResponseChunkSerializer


class ClaudeProvider(LLMProvider):

    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.request_serializer = ClaudeRequestSerializer()
        self.response_serializer = ClaudeResponseSerializer()
        self.response_chunk_serializer = ClaudeResponseChunkSerializer()

    def generate_response(self, request: LLMRequest) -> LLMResponse:

        
        payload = self.request_serializer.serialize(request)

        client_response = self.client.messages.create(**payload)

        llm_response = self.response_serializer.serialize(client_response)
    
        return self._finalize_response(request,llm_response)
    

    def _finalize_response(self,request: LLMRequest, response: LLMResponse)-> LLMResponse:

        while response.tool_calls:


            tool_results = self._execute_tools(response.tool_calls)

            payload = self.request_serializer.serialize_tool_results(
                request=request,
                assistant_response=response.raw_response,
                tool_results=tool_results,
            )

            response = self.response_serializer.serialize(
                self.client.messages.create(**payload)
            )

        return response
    

    def generate_stream(
    self,
    request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:
        
        payload = self.request_serializer.serialize(request)

        payload["stream"] = True

        stream = self.client.messages.create(**payload)

            
        for event in stream:
            print("=" * 80)
            print(type(event))
            #print("-" * 80)

            if hasattr(event, "delta"):
                print(type(event.delta))
                print(event.delta)
                print(event.content_block)
            continue

            chunk = self.response_chunk_serializer.serialize(event)
            if chunk:
                yield chunk    
