from openai import OpenAI
from typing import Generator

from config import OPENAI_API_KEY
from providers import LLMProvider
from models import LLMResponse,LLMRequest,LLMResponseChunk
from serializers import OpenAIRequestSerializer,OpenAIResponseSerializer,OpenAIResponseChunkSerializer


class OpenAIProvider(LLMProvider):

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.request_serializer = OpenAIRequestSerializer()
        self.response_serializer = OpenAIResponseSerializer()
        self.response_chunk_serializer = OpenAIResponseChunkSerializer()

    # Generate a complete response for the given prompt.
    def generate(self, request: LLMRequest) -> LLMResponse:
        
        payload = self.request_serializer.serialize(request)

        response = self.client.responses.create(**payload)

        return self.response_serializer.serialize(response)
    
    
    # Generate a stream of response for the given prompt.
    def generate_stream(
    self,
    request: LLMRequest,
    )-> Generator[LLMResponseChunk, None, None]:
        
        payload = self.request_serializer.serialize(request)
         
        payload["stream"] = True
         
        stream = self.client.responses.create(**payload)

        for event in stream:
            chunk = self.response_chunk_serializer.serialize(event)
            if chunk:
                yield chunk
    
    