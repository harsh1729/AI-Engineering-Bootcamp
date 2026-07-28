from unittest.mock import MagicMock

from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import (
    EmbeddingBatchResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingVector,
)
from app.embeddings.embedding_service import EmbeddingService

MODEL = "text-embedding-3-small"


def _single_response() -> EmbeddingResponse:
    return EmbeddingResponse(
        embedding=[0.1, 0.2],
        model=MODEL,
        usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2),
    )


def _batch_response() -> EmbeddingBatchResponse:
    return EmbeddingBatchResponse(
        embeddings=[
            EmbeddingVector(embedding=[0.1], model=MODEL),
            EmbeddingVector(embedding=[0.2], model=MODEL),
        ],
        usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
    )


class TestEmbeddingServiceDelegation:
    def test_embed_delegates_to_injected_provider(self) -> None:
        provider = MagicMock(spec=BaseEmbeddingProvider)
        expected = _single_response()
        provider.embed.return_value = expected
        service = EmbeddingService(provider)

        response = service.embed("hello")

        provider.embed.assert_called_once_with("hello")
        assert response is expected

    def test_embed_batch_delegates_unchanged(self) -> None:
        provider = MagicMock(spec=BaseEmbeddingProvider)
        expected = _batch_response()
        provider.embed_batch.return_value = expected
        service = EmbeddingService(provider)

        response = service.embed_batch(["one", "two"])

        provider.embed_batch.assert_called_once_with(["one", "two"])
        assert response is expected


class TestEmbeddingServiceProviderAgnostic:
    def test_swapping_provider_changes_behavior(self) -> None:
        first_provider = MagicMock(spec=BaseEmbeddingProvider)
        second_provider = MagicMock(spec=BaseEmbeddingProvider)
        first_provider.embed_batch.return_value = EmbeddingBatchResponse(
            embeddings=[EmbeddingVector(embedding=[1.0], model="model-a")],
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        second_provider.embed_batch.return_value = EmbeddingBatchResponse(
            embeddings=[EmbeddingVector(embedding=[2.0], model="model-b")],
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )

        first_service = EmbeddingService(first_provider)
        second_service = EmbeddingService(second_provider)

        first_batch = first_service.embed_batch(["text"])
        second_batch = second_service.embed_batch(["text"])

        assert first_batch.embeddings[0].embedding == [1.0]
        assert second_batch.embeddings[0].embedding == [2.0]
        assert first_batch is not second_batch

    def test_service_accepts_any_base_embedding_provider_implementation(self) -> None:
        provider = MagicMock(spec=BaseEmbeddingProvider)
        provider.embed.return_value = _single_response()
        service = EmbeddingService(provider)

        response = service.embed("chunk text")

        assert isinstance(response, EmbeddingResponse)
        provider.embed.assert_called_once_with("chunk text")
