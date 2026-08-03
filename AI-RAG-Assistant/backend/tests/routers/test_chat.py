from unittest.mock import MagicMock, create_autospec

import pytest
from fastapi.testclient import TestClient
from llm_sdk.enums import MessageRole

from app.main import app
from app.models.chat import ChatResponse
from app.services.chat_service import ChatService

client = TestClient(app)


@pytest.fixture
def chat_service_override() -> MagicMock:
    return create_autospec(ChatService, instance=True)


@pytest.fixture
def client_with_chat_service(chat_service_override: MagicMock) -> TestClient:
    from app.dependencies import get_chat_service

    app.dependency_overrides[get_chat_service] = lambda: chat_service_override
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestChatEndpoint:
    def test_chat_delegates_to_chat_service(
        self,
        client_with_chat_service: TestClient,
        chat_service_override: MagicMock,
    ) -> None:
        chat_service_override.chat.return_value = ChatResponse(
            response="Service answer",
            warning=None,
        )

        response = client_with_chat_service.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert response.json() == {"response": "Service answer", "warning": None, "sources": []}
        chat_service_override.chat.assert_called_once()
        chat_request = chat_service_override.chat.call_args.args[0]
        assert chat_request.messages[0].role == MessageRole.USER

    def test_chat_returns_400_when_chat_service_raises_value_error(
        self,
        client_with_chat_service: TestClient,
        chat_service_override: MagicMock,
    ) -> None:
        chat_service_override.chat.side_effect = ValueError(
            "At least one user message is required."
        )

        response = client_with_chat_service.post(
            "/chat",
            json={
                "guest_id": "guest-123",
                "messages": [{"role": "assistant", "content": "Hello"}],
                "document_ids": ["doc-1"],
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "At least one user message is required."


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
    ) -> None:
        from llm_sdk.providers.llm_provider import LLMProvider

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

        response = client.post(
            "/chat/stream",
            json={
                "guest_id": "guest-stream-error",
                "messages": [{"role": "user", "content": "Hello"}],
                "provider": "gemini",
                "model": "gemini-3.5-pro",
            },
        )

        assert response.status_code == 200
        assert "Model not found." in response.text
        assert "The assistant could not respond" not in response.text


class TestChatStreamTools:
    def test_stream_passes_tools_to_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
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

        response = client.post(
            "/chat/stream",
            json={
                "guest_id": "guest-stream-tools",
                "messages": [{"role": "user", "content": "Weather in London?"}],
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
        )

        assert response.status_code == 200
        assert response.text == "Sunny in London"
        assert len(captured_requests) == 1
        assert captured_requests[0].tools == DEFAULT_CHAT_TOOLS

    def test_stream_includes_tools_for_gemini(
        self,
        monkeypatch: pytest.MonkeyPatch,
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

        response = client.post(
            "/chat/stream",
            json={
                "guest_id": "guest-stream-gemini",
                "messages": [{"role": "user", "content": "Weather in London?"}],
                "provider": "gemini",
                "model": "gemini-2.0-flash",
            },
        )

        assert response.status_code == 200
        assert captured_requests[0].tools == DEFAULT_CHAT_TOOLS

