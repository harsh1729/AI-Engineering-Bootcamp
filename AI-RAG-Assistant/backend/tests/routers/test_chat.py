import uuid
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from fastapi.testclient import TestClient
from llm_sdk.enums import MessageRole

from app.config import GUEST_USAGE_LIMIT_MESSAGE
from app.database.deps import (
    get_chat_persistence_service,
    get_document_access_service,
    get_rag_options_resolver_service,
    get_usage_tracker_service,
)
from app.context.context_models import ContextSource
from app.main import app
from app.models.chat import ChatResponse
from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)
from app.services.chat_service import ChatService, PreparedStreamRequest
from app.services.document_access_service import DocumentAccessDeniedError
from app.services.rag_options_resolver import MixedRagConfigError
from app.services.usage_exceptions import UsageLimitExceeded

CHAT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


@pytest.fixture
def chat_service_override() -> MagicMock:
    return create_autospec(ChatService, instance=True)


@pytest.fixture
def persistence_override() -> MagicMock:
    mock = MagicMock()
    chat = MagicMock()
    chat.id = CHAT_ID
    mock.get_or_create_chat = AsyncMock(return_value=chat)
    mock.append_user_message = AsyncMock()
    mock.append_assistant_message = AsyncMock()
    return mock


@pytest.fixture
def usage_tracker_override() -> MagicMock:
    mock = MagicMock()
    mock.record_guest_interaction = AsyncMock(return_value=1)
    mock.record_user_interaction = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def document_access_override() -> MagicMock:
    mock = MagicMock()
    mock.validate_document_ids = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def rag_options_resolver_override() -> MagicMock:
    mock = MagicMock()
    mock.resolve_rag_options_for_documents = AsyncMock(return_value=RagOptions())
    return mock


@pytest.fixture
def client_with_overrides(
    chat_service_override: MagicMock,
    persistence_override: MagicMock,
    usage_tracker_override: MagicMock,
    document_access_override: MagicMock,
    rag_options_resolver_override: MagicMock,
) -> TestClient:
    from app.dependencies import get_chat_service

    app.dependency_overrides[get_chat_service] = lambda: chat_service_override
    app.dependency_overrides[get_chat_persistence_service] = lambda: persistence_override
    app.dependency_overrides[get_usage_tracker_service] = lambda: usage_tracker_override
    app.dependency_overrides[get_document_access_service] = lambda: document_access_override
    app.dependency_overrides[get_rag_options_resolver_service] = (
        lambda: rag_options_resolver_override
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestChatEndpoint:
    def test_chat_delegates_to_chat_service_and_persists_messages(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        persistence_override: MagicMock,
    ) -> None:
        chat_service_override.chat.return_value = ChatResponse(
            response="Service answer",
            warning=None,
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "response": "Service answer",
            "warning": None,
            "sources": [],
            "chat_id": str(CHAT_ID),
        }
        chat_service_override.chat.assert_called_once()
        chat_request = chat_service_override.chat.call_args.args[0]
        assert chat_request.messages[0].role == MessageRole.USER
        persistence_override.get_or_create_chat.assert_awaited_once()
        persistence_override.append_user_message.assert_awaited_once()
        persistence_override.append_assistant_message.assert_awaited_once()

    def test_chat_returns_403_when_document_not_accessible(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        persistence_override: MagicMock,
        document_access_override: MagicMock,
    ) -> None:
        document_access_override.validate_document_ids = AsyncMock(
            side_effect=DocumentAccessDeniedError(
                "Document '11111111-2222-3333-4444-555555555555' was not found."
            ),
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
                "document_ids": ["11111111-2222-3333-4444-555555555555"],
            },
        )

        assert response.status_code == 403
        assert "was not found" in response.json()["detail"]
        chat_service_override.chat.assert_not_called()
        persistence_override.get_or_create_chat.assert_not_awaited()

    def test_chat_resolves_rag_options_from_attached_documents(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        rag_options_resolver_override: MagicMock,
    ) -> None:
        resolved = RagOptions(
            chunking_strategy=ChunkingStrategy.CHARACTER,
            embedding_provider=EmbeddingProviderType.VOYAGE,
            vector_store=VectorStoreType.CHROMA,
        )
        rag_options_resolver_override.resolve_rag_options_for_documents = AsyncMock(
            return_value=resolved,
        )
        chat_service_override.chat.return_value = ChatResponse(
            response="Grounded answer",
            warning=None,
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
                "document_ids": ["11111111-2222-3333-4444-555555555555"],
            },
        )

        assert response.status_code == 200
        rag_options_resolver_override.resolve_rag_options_for_documents.assert_awaited_once_with(
            ["11111111-2222-3333-4444-555555555555"],
        )
        chat_request = chat_service_override.chat.call_args.args[0]
        assert chat_request.rag_options == resolved

    def test_chat_returns_400_when_attached_documents_have_mixed_rag_config(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        rag_options_resolver_override: MagicMock,
    ) -> None:
        rag_options_resolver_override.resolve_rag_options_for_documents = AsyncMock(
            side_effect=MixedRagConfigError(
                "Attached documents use different indexing settings. "
                "Remove documents or re-upload with matching RAG options."
            ),
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
                "document_ids": [
                    "11111111-2222-3333-4444-555555555555",
                    "22222222-3333-4444-5555-666666666666",
                ],
            },
        )

        assert response.status_code == 400
        assert "different indexing settings" in response.json()["detail"]
        chat_service_override.chat.assert_not_called()

    def test_chat_returns_400_when_chat_service_raises_value_error(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        persistence_override: MagicMock,
    ) -> None:
        chat_service_override.chat.side_effect = ValueError(
            "At least one user message is required."
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "messages": [{"role": "assistant", "content": "Hello"}],
                "document_ids": ["doc-1"],
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "At least one user message is required."
        persistence_override.append_assistant_message.assert_not_called()

    def test_chat_returns_403_when_guest_usage_limit_reached(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        persistence_override: MagicMock,
        usage_tracker_override: MagicMock,
    ) -> None:
        usage_tracker_override.record_guest_interaction = AsyncMock(
            side_effect=UsageLimitExceeded(GUEST_USAGE_LIMIT_MESSAGE),
        )

        response = client_with_overrides.post(
            "/chat",
            json={
                "guest_id": "guest-limit",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 403
        assert response.json()["detail"] == GUEST_USAGE_LIMIT_MESSAGE
        chat_service_override.chat.assert_not_called()
        persistence_override.get_or_create_chat.assert_not_awaited()

    def test_chat_stream_returns_403_when_guest_usage_limit_reached(
        self,
        client_with_overrides: TestClient,
        persistence_override: MagicMock,
        usage_tracker_override: MagicMock,
    ) -> None:
        usage_tracker_override.record_guest_interaction = AsyncMock(
            side_effect=UsageLimitExceeded(GUEST_USAGE_LIMIT_MESSAGE),
        )

        response = client_with_overrides.post(
            "/chat/stream",
            json={
                "guest_id": "guest-limit",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 403
        assert response.json()["detail"] == GUEST_USAGE_LIMIT_MESSAGE
        persistence_override.get_or_create_chat.assert_not_awaited()


class TestChatStreamRag:
    def test_chat_stream_resolves_rag_options_and_sets_sources_header(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        rag_options_resolver_override: MagicMock,
    ) -> None:
        resolved = RagOptions(
            chunking_strategy=ChunkingStrategy.CHARACTER,
            embedding_provider=EmbeddingProviderType.VOYAGE,
            vector_store=VectorStoreType.CHROMA,
        )
        rag_options_resolver_override.resolve_rag_options_for_documents = AsyncMock(
            return_value=resolved,
        )

        rag_service = MagicMock()
        rag_service.stream_answer.return_value = iter(["Grounded ", "answer"])
        sources = [
            ContextSource(
                source_number=1,
                document_id="11111111-2222-3333-4444-555555555555",
                filename="notes.pdf",
            )
        ]
        chat_service_override.prepare_stream.return_value = PreparedStreamRequest(
            llm_request=MagicMock(),
            sources=sources,
            rag_service=rag_service,
        )

        response = client_with_overrides.post(
            "/chat/stream",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Summarize the doc"}],
                "document_ids": ["11111111-2222-3333-4444-555555555555"],
            },
        )

        assert response.status_code == 200
        assert response.text == "Grounded answer"
        assert response.headers["X-Chat-Id"] == str(CHAT_ID)
        assert "notes.pdf" in response.headers["X-RAG-Sources"]
        rag_options_resolver_override.resolve_rag_options_for_documents.assert_awaited_once()
        chat_service_override.prepare_stream.assert_called_once()

    def test_chat_stream_returns_403_when_document_not_accessible(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        document_access_override: MagicMock,
    ) -> None:
        document_access_override.validate_document_ids = AsyncMock(
            side_effect=DocumentAccessDeniedError(
                "Document '11111111-2222-3333-4444-555555555555' was not found."
            ),
        )

        response = client_with_overrides.post(
            "/chat/stream",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
                "document_ids": ["11111111-2222-3333-4444-555555555555"],
            },
        )

        assert response.status_code == 403
        chat_service_override.prepare_stream.assert_not_called()

    def test_chat_stream_returns_400_for_mixed_rag_config(
        self,
        client_with_overrides: TestClient,
        chat_service_override: MagicMock,
        rag_options_resolver_override: MagicMock,
    ) -> None:
        rag_options_resolver_override.resolve_rag_options_for_documents = AsyncMock(
            side_effect=MixedRagConfigError(
                "Attached documents use different indexing settings."
            ),
        )

        response = client_with_overrides.post(
            "/chat/stream",
            json={
                "guest_id": "guest-123",
                "provider": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
                "document_ids": [
                    "11111111-2222-3333-4444-555555555555",
                    "22222222-3333-4444-5555-666666666666",
                ],
            },
        )

        assert response.status_code == 400
        assert "different indexing settings" in response.json()["detail"]
        chat_service_override.prepare_stream.assert_not_called()


class TestStreamingErrorMessages:
    def test_extracts_gemini_not_found_message(self) -> None:
        from app.routers.chat import _extract_provider_error_message

        exc = Exception(
            "404 NOT_FOUND. {'error': {'code': 404, "
            "'message': 'models/gemini-3.5-pro is not found for API version v1beta.', "
            "'status': 'NOT_FOUND'}}"
        )

        message = _extract_provider_error_message(exc)

        assert message == "models/gemini-3.5-pro is not found for API version v1beta."

    def test_extracts_gemini_high_demand_message(self) -> None:
        from app.routers.chat import _extract_provider_error_message

        exc = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, "
            "'message': 'This model is currently experiencing high demand.', "
            "'status': 'UNAVAILABLE'}}"
        )

        message = _extract_provider_error_message(exc)

        assert message == "This model is currently experiencing high demand."

    def test_stream_yields_provider_error_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        persistence_override: MagicMock,
        usage_tracker_override: MagicMock,
    ) -> None:
        from llm_sdk.providers.llm_provider import LLMProvider

        from app.dependencies import get_chat_service
        from app.routers import chat as chat_router

        class FailingProvider(LLMProvider):
            def generate_response(self, request):
                raise NotImplementedError

            def generate_stream(self, request):
                raise Exception(
                    "404 NOT_FOUND. {'error': {'message': 'Model not found.'}}"
                )
                yield  # pragma: no cover

        monkeypatch.setattr(
            chat_router.ProviderFactory,
            "create",
            lambda provider: FailingProvider(),
        )

        app.dependency_overrides[get_chat_persistence_service] = (
            lambda: persistence_override
        )
        app.dependency_overrides[get_usage_tracker_service] = (
            lambda: usage_tracker_override
        )
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/chat/stream",
                    json={
                        "guest_id": "guest-stream-error",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "provider": "gemini",
                        "model": "gemini-3.5-pro",
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert "Model not found." in response.text
        assert "The assistant could not respond" not in response.text
        assert response.headers["X-Chat-Id"] == str(CHAT_ID)
        persistence_override.append_assistant_message.assert_awaited_once()


class TestChatStreamTools:
    def test_stream_passes_tools_to_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        persistence_override: MagicMock,
        usage_tracker_override: MagicMock,
    ) -> None:
        from llm_sdk.models import LLMResponseChunk
        from llm_sdk.providers.llm_provider import LLMProvider

        from app.llm.chat_tools import DEFAULT_CHAT_TOOLS
        from app.routers import chat as chat_router

        captured_requests: list = []

        class CapturingProvider(LLMProvider):
            def generate_response(self, request):
                raise NotImplementedError

            def generate_stream(self, request):
                captured_requests.append(request)
                yield LLMResponseChunk(text="Sunny in London")

        monkeypatch.setattr(
            chat_router.ProviderFactory,
            "create",
            lambda provider: CapturingProvider(),
        )

        app.dependency_overrides[get_chat_persistence_service] = (
            lambda: persistence_override
        )
        app.dependency_overrides[get_usage_tracker_service] = (
            lambda: usage_tracker_override
        )
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/chat/stream",
                    json={
                        "guest_id": "guest-stream-tools",
                        "messages": [{"role": "user", "content": "Weather in London?"}],
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.text == "Sunny in London"
        assert response.headers["X-Chat-Id"] == str(CHAT_ID)
        assert len(captured_requests) == 1
        assert captured_requests[0].tools == DEFAULT_CHAT_TOOLS

    def test_stream_includes_tools_for_gemini(
        self,
        monkeypatch: pytest.MonkeyPatch,
        persistence_override: MagicMock,
        usage_tracker_override: MagicMock,
    ) -> None:
        from llm_sdk.models import LLMResponseChunk
        from llm_sdk.providers.llm_provider import LLMProvider

        from app.llm.chat_tools import DEFAULT_CHAT_TOOLS
        from app.routers import chat as chat_router

        captured_requests: list = []

        class CapturingProvider(LLMProvider):
            def generate_response(self, request):
                raise NotImplementedError

            def generate_stream(self, request):
                captured_requests.append(request)
                yield LLMResponseChunk(text="Plain answer")

        monkeypatch.setattr(
            chat_router.ProviderFactory,
            "create",
            lambda provider: CapturingProvider(),
        )

        app.dependency_overrides[get_chat_persistence_service] = (
            lambda: persistence_override
        )
        app.dependency_overrides[get_usage_tracker_service] = (
            lambda: usage_tracker_override
        )
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/chat/stream",
                    json={
                        "guest_id": "guest-stream-gemini",
                        "messages": [{"role": "user", "content": "Weather in London?"}],
                        "provider": "gemini",
                        "model": "gemini-2.0-flash",
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert captured_requests[0].tools == DEFAULT_CHAT_TOOLS
