from models import LLMMessage
from enums import MessageRole

from serializers import LLMMessageSerializer


class ClaudeMessageSerializer(LLMMessageSerializer):

    def serialize_messages(
        self,
        messages: list[LLMMessage],
    ) -> dict:

        system_messages = []
        claude_messages = []

        for message in messages:

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
            "system": "\n\n".join(system_messages) if system_messages else None,
            "messages": claude_messages,
        }