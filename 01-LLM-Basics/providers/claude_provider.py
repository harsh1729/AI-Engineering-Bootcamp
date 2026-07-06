from anthropic import Anthropic
from typing import Generator

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

    def generate(self, request: LLMRequest) -> LLMResponse:

        
        payload = self.request_serializer.serialize(request)

        response = self.client.messages.create(**payload)

        return self.response_serializer.serialize(response)
    

    def generate_stream(
    self,
    request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:
        
        payload = self.request_serializer.serialize(request)

        payload["stream"] = True

        stream = self.client.messages.create(**payload)

            
        for event in stream:
            # print("=" * 80)
            # print(type(event))
            # print("-" * 80)

            chunk = self.response_chunk_serializer.serialize(event)
            if chunk:
                yield chunk    
