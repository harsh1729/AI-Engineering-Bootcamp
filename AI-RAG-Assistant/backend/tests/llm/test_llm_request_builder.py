from llm_sdk.enums import MessageRole, ProviderType
from llm_sdk.models import LLMMessage

from app.llm.chat_tools import DEFAULT_CHAT_TOOLS
from app.llm.llm_request_builder import TOOLS_SYSTEM_PROMPT, build_chat_llm_request
from app.models.chat import ChatMessage, ChatRequest


class TestBuildChatLLMRequest:
    def test_includes_tools_and_system_prompt_when_enabled(self) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Weather in London?")],
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        llm_request = build_chat_llm_request(request, include_tools=True)

        assert llm_request.tools == DEFAULT_CHAT_TOOLS
        assert llm_request.messages[0] == LLMMessage(
            role=MessageRole.SYSTEM,
            content=TOOLS_SYSTEM_PROMPT,
        )
        assert llm_request.messages[1].content == "Weather in London?"

    def test_omits_tools_when_not_requested(self) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        llm_request = build_chat_llm_request(request, include_tools=False)

        assert llm_request.tools == []

    def test_preserves_existing_system_message(self) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="Custom system prompt"),
                ChatMessage(role=MessageRole.USER, content="Hello"),
            ],
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        llm_request = build_chat_llm_request(request, include_tools=True)

        assert llm_request.tools == DEFAULT_CHAT_TOOLS
        assert len(llm_request.messages) == 2
        assert llm_request.messages[0].content == "Custom system prompt"

    def test_includes_tools_for_gemini(self) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Weather in London?")],
            provider=ProviderType.GEMINI,
            model="gemini-2.0-flash",
        )

        llm_request = build_chat_llm_request(request, include_tools=True)

        assert llm_request.tools == DEFAULT_CHAT_TOOLS
        assert llm_request.messages[0] == LLMMessage(
            role=MessageRole.SYSTEM,
            content=TOOLS_SYSTEM_PROMPT,
        )
        assert llm_request.messages[1].content == "Weather in London?"
