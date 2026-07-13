from google.genai import types

from config import GEMINI_MODEL, GEMINI_THINKING_LEVEL
from enums import MessageRole
from models import LLMMessage, LLMRequest
from serializers import BaseRequestSerializer


class GeminiRequestSerializer(BaseRequestSerializer):

    def serialize(
        self,
        request: LLMRequest,
        model: str | None = None,
    ) -> dict:

        return {
            "model": model or GEMINI_MODEL,
            "contents": self._serialize_contents(request.messages),
            "config": types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                system_instruction=self._serialize_system_messages(request.messages),
                thinking_config=types.ThinkingConfig(
                    thinking_level=GEMINI_THINKING_LEVEL.value,
                ),
            ),
        }

    def _serialize_contents(
        self,
        messages: list[LLMMessage],
    ) -> list[types.Content]:

        contents : list[types.Content] = []
        

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                continue

            contents.append(
                types.Content(
                    role=self._serialize_role(message.role),
                    parts=[types.Part(text=message.content)],
                )
            )

        return contents

    def _serialize_role(
        self,
        role: MessageRole,
    ) -> str:

        if role == MessageRole.ASSISTANT:
            return "model"

        return role.value

    def _serialize_system_messages(
        self,
        messages: list[LLMMessage],
    ) -> str | None:

        system_messages = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.content)

        return "\n\n".join(system_messages) if system_messages else None
