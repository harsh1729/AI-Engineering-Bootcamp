from models import LLMMessage,LLMRequest

from serializers import BaseRequestSerializer
from config import OPENAI_MODEL


class OpenAIRequestSerializer(BaseRequestSerializer):
    
    def serialize(
    self,
    request: LLMRequest,
    ) -> dict:
        return {
            "model": OPENAI_MODEL,
            "input": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }