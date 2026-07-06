from models import LLMRequest
from enums import MessageRole

from serializers import BaseRequestSerializer
from config import ANTHROPIC_MODEL


class ClaudeRequestSerializer(BaseRequestSerializer):

    def serialize(
        self,
        request: LLMRequest,
    ) -> dict:

        system_messages = []
        claude_messages = []

        for message in request.messages:

            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.content)

            else:
                claude_messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        return {

            "model": ANTHROPIC_MODEL,
            "max_tokens": request.max_tokens,
            "system": "\n\n".join(system_messages) if system_messages else None,
            "messages": claude_messages,
        }