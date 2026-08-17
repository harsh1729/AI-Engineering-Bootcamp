from unittest.mock import MagicMock, create_autospec

import pytest
from llm_sdk.enums import MessageRole, ProviderType
from llm_sdk.models import LLMMessage, LLMResponse
from llm_sdk.providers.llm_provider import LLMProvider

from app.context.context_models import ContextSource
from app.llm.chat_tools import DEFAULT_CHAT_TOOLS
from app.llm.llm_request_builder import TOOLS_SYSTEM_PROMPT
from app.models.chat import ChatMessage, ChatRequest, ChatResponse
from app.models.rag import RAGRequest, RAGResponse
from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)
from app.services.chat_service import ChatService, PreparedStreamRequest
from app.services.rag_service import RAGService

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def rag_service() -> MagicMock:
    return create_autospec(RAGService, instance=True)


@pytest.fixture
def llm_provider() -> MagicMock:
    return create_autospec(LLMProvider, instance=True)


@pytest.fixture
def chat_service(rag_service: MagicMock, llm_provider: MagicMock) -> ChatService:
    return ChatService(rag_service=rag_service, llm_provider=llm_provider)


class TestChatServiceDirectChat:
    def test_routes_to_llm_when_document_ids_are_empty(
        self,
        chat_service: ChatService,
        llm_provider: MagicMock,
        rag_service: MagicMock,
    ) -> None:
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-1",
            provider="openai",
            model="gpt-4o-mini",
            text="Direct answer",
            finish_reason="stop",
        )
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        response = chat_service.chat(request)

        rag_service.ask.assert_not_called()
        llm_provider.generate_response.assert_called_once()
        llm_request = llm_provider.generate_response.call_args.args[0]
        assert llm_request.tools == DEFAULT_CHAT_TOOLS
        assert llm_request.messages[0] == LLMMessage(
            role=MessageRole.SYSTEM,
            content=TOOLS_SYSTEM_PROMPT,
        )
        assert llm_request.messages[1] == LLMMessage(role=MessageRole.USER, content="Hello")
        assert llm_request.provider == ProviderType.OPENAI
        assert llm_request.model == "gpt-4o-mini"
        assert response == ChatResponse(response="Direct answer", warning=None)

    def test_forwards_full_conversation_to_llm(
        self,
        chat_service: ChatService,
        llm_provider: MagicMock,
    ) -> None:
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-2",
            provider="gemini",
            model="gemini-2.0-flash",
            text="Follow-up answer",
            finish_reason="stop",
        )
        request = ChatRequest(
            guest_id="guest-1",
            messages=[
                ChatMessage(role=MessageRole.USER, content="First"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Reply"),
                ChatMessage(role=MessageRole.USER, content="Second"),
            ],
        )

        chat_service.chat(request)

        llm_request = llm_provider.generate_response.call_args.args[0]
        assert len(llm_request.messages) == 4
        assert llm_request.tools == DEFAULT_CHAT_TOOLS

    def test_includes_tools_for_gemini_direct_chat(
        self,
        chat_service: ChatService,
        llm_provider: MagicMock,
    ) -> None:
        llm_provider.generate_response.return_value = LLMResponse(
            id="resp-gemini",
            provider="gemini",
            model="gemini-2.0-flash",
            text="Plain answer",
            finish_reason="stop",
        )
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Weather?")],
            provider=ProviderType.GEMINI,
            model="gemini-2.0-flash",
        )

        chat_service.chat(request)

        llm_request = llm_provider.generate_response.call_args.args[0]
        assert llm_request.tools == DEFAULT_CHAT_TOOLS


class TestChatServiceRagChat:
    def test_routes_to_rag_when_document_ids_are_present(
        self,
        chat_service: ChatService,
        llm_provider: MagicMock,
        rag_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "app.services.chat_service.build_rag_service",
            lambda rag_options, provider, context_builder: rag_service,
        )
        rag_service.ask.return_value = RAGResponse(
            answer="Grounded answer",
            sources=[
                ContextSource(
                    source_number=1,
                    document_id=DOCUMENT_ID,
                    filename="notes.pdf",
                )
            ],
            warning=None,
        )
        stored_options = RagOptions(
            chunking_strategy=ChunkingStrategy.CHARACTER,
            embedding_provider=EmbeddingProviderType.VOYAGE,
            vector_store=VectorStoreType.CHROMA,
        )
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="What is RAG?")],
            document_ids=[DOCUMENT_ID],
            rag_options=stored_options,
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        response = chat_service.chat(request)

        llm_provider.generate_response.assert_not_called()
        rag_service.ask.assert_called_once_with(
            RAGRequest(
                query="What is RAG?",
                document_ids=[DOCUMENT_ID],
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
            )
        )
        assert response == ChatResponse(
            response="Grounded answer",
            warning=None,
            sources=[
                ContextSource(
                    source_number=1,
                    document_id=DOCUMENT_ID,
                    filename="notes.pdf",
                )
            ],
        )

    def test_requires_resolved_rag_options_for_document_chat(
        self,
        chat_service: ChatService,
    ) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="What is RAG?")],
            document_ids=[DOCUMENT_ID],
        )

        with pytest.raises(ValueError, match="RAG options must be resolved"):
            chat_service.chat(request)

    def test_requires_user_message_for_rag_requests(
        self,
        chat_service: ChatService,
    ) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.ASSISTANT, content="Hello")],
            document_ids=[DOCUMENT_ID],
            rag_options=RagOptions(),
        )

        with pytest.raises(ValueError, match="At least one user message is required."):
            chat_service.chat(request)


class TestChatServicePrepareStream:
    def test_prepare_stream_builds_direct_llm_request_without_documents(
        self,
        chat_service: ChatService,
    ) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        prepared = chat_service.prepare_stream(request)

        assert prepared.sources == []
        assert prepared.rag_service is None
        assert prepared.llm_request.provider == ProviderType.OPENAI

    def test_prepare_stream_builds_rag_request_when_documents_attached(
        self,
        chat_service: ChatService,
        rag_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.context.context_models import ContextSource
        from app.services import chat_service as chat_service_module
        from app.services.rag_service import PreparedRagContext

        stored_options = RagOptions(
            chunking_strategy=ChunkingStrategy.CHARACTER,
            embedding_provider=EmbeddingProviderType.VOYAGE,
            vector_store=VectorStoreType.CHROMA,
        )
        llm_request = MagicMock()
        sources = [
            ContextSource(
                source_number=1,
                document_id=DOCUMENT_ID,
                filename="notes.pdf",
            )
        ]
        rag_service.prepare.return_value = PreparedRagContext(
            llm_request=llm_request,
            sources=sources,
            retrieval_request=MagicMock(),
            retrieval_response=MagicMock(),
        )
        monkeypatch.setattr(
            chat_service_module,
            "build_rag_service",
            lambda rag_options, provider, context_builder: rag_service,
        )

        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="What is RAG?")],
            document_ids=[DOCUMENT_ID],
            rag_options=stored_options,
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )

        prepared = chat_service.prepare_stream(request)

        assert prepared.llm_request is llm_request
        assert prepared.sources == sources
        assert prepared.rag_service is rag_service
        rag_service.prepare.assert_called_once()

    def test_prepare_stream_requires_resolved_rag_options(
        self,
        chat_service: ChatService,
    ) -> None:
        request = ChatRequest(
            guest_id="guest-1",
            messages=[ChatMessage(role=MessageRole.USER, content="What is RAG?")],
            document_ids=[DOCUMENT_ID],
        )

        with pytest.raises(ValueError, match="RAG options must be resolved"):
            chat_service.prepare_stream(request)
