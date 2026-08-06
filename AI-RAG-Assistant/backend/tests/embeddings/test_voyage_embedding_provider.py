from unittest.mock import MagicMock

import pytest

from app.embeddings.embedding_exceptions import EmbeddingProviderError, EmbeddingRateLimitError
from app.embeddings.embedding_models import EmbeddingUsage, EmbeddingVector
from app.embeddings.voyage_embedding_provider import VoyageEmbeddingProvider
from voyageai.error import RateLimitError as VoyageRateLimitError, VoyageError

MODEL = "voyage-4-lite"


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def provider(mock_client: MagicMock) -> VoyageEmbeddingProvider:
    return VoyageEmbeddingProvider(client=mock_client, model=MODEL)


def _make_voyage_response(
    *,
    embeddings: list[list[float]],
    total_tokens: int = 8,
) -> MagicMock:
    response = MagicMock()
    response.embeddings = embeddings
    response.total_tokens = total_tokens
    return response


class TestVoyageEmbeddingProviderEmbed:
    def test_single_embedding_uses_query_input_type(
        self, provider: VoyageEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.return_value = _make_voyage_response(embeddings=[[0.1, 0.2, 0.3]])

        response = provider.embed("hello world")

        mock_client.embed.assert_called_once_with(
            ["hello world"],
            model=MODEL,
            input_type="query",
        )
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.model == MODEL
        assert response.usage.total_tokens == 8


class TestVoyageEmbeddingProviderEmbedBatch:
    def test_batch_embedding_uses_document_input_type(
        self, provider: VoyageEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.return_value = _make_voyage_response(
            embeddings=[[0.1], [0.2], [0.3]]
        )

        batch = provider.embed_batch(["alpha", "beta", "gamma"])

        mock_client.embed.assert_called_once_with(
            ["alpha", "beta", "gamma"],
            model=MODEL,
            input_type="document",
        )
        assert len(batch.embeddings) == 3
        assert [vector.embedding for vector in batch.embeddings] == [[0.1], [0.2], [0.3]]

    def test_empty_input_returns_synthetic_zero_usage_without_api_call(
        self, provider: VoyageEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        batch = provider.embed_batch([])

        assert batch.embeddings == []
        assert batch.usage == EmbeddingUsage(prompt_tokens=0, total_tokens=0)
        mock_client.embed.assert_not_called()


class TestVoyageEmbeddingProviderErrors:
    def test_rate_limit_error_is_converted(
        self, provider: VoyageEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.side_effect = VoyageRateLimitError("rate limited")

        with pytest.raises(EmbeddingRateLimitError, match="rate limited"):
            provider.embed_batch(["hello"])

    def test_voyage_error_is_converted(
        self, provider: VoyageEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.side_effect = VoyageError("server error")

        with pytest.raises(EmbeddingProviderError, match="server error"):
            provider.embed_batch(["hello"])
