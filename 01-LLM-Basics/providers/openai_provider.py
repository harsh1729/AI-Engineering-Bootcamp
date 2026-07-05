from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
from providers import LLMProvider
from models import LLMResponse,LLMRequest
from enums import ProviderType
from serializers import OpenAIMessageSerializer


class OpenAIProvider(LLMProvider):

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.serializer = OpenAIMessageSerializer()

    def generate(self, request: LLMRequest) -> LLMResponse:

        serializer = OpenAIMessageSerializer()

        messages = serializer.serialize_messages(
            request.messages
        )
        
        response = self.client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        return LLMResponse(
            id=response.id,
            provider=ProviderType.OPENAI,
            model=response.model,
            role=response.output[0].role,
            text=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=response.status,
            tool_calls=[],
            reasoning=None,
            raw_response=response,  
        )
    
    