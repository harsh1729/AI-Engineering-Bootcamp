from models import LLMMessage

from serializers import LLMMessageSerializer


class OpenAIMessageSerializer(LLMMessageSerializer):

    def serialize_messages(
        self,
        messages: list[LLMMessage],
    ) -> list[dict]:

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]