from unittest.mock import MagicMock

import pytest
from cohere.core.api_error import ApiError as CohereApiError
from cohere import TooManyRequestsError as CohereTooManyRequestsError

from app.embeddings.cohere_embedding_provider import (
    COHERE_EMBED_BATCH_SIZE,
    CohereEmbeddingProvider,
)
from app.embeddings.embedding_exceptions import EmbeddingProviderError, EmbeddingRateLimitError
from app.embeddings.embedding_models import EmbeddingUsage

MODEL = "embed-v4.0"


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def provider(mock_client: MagicMock) -> CohereEmbeddingProvider:
    return CohereEmbeddingProvider(client=mock_client, model=MODEL)


def _make_cohere_response(
    *,
    embeddings: list[list[float]],
    input_tokens: int = 12,
) -> MagicMock:
    response = MagicMock()
    response.embeddings.float = embeddings
    response.meta.billed_units.input_tokens = input_tokens
    response.meta.tokens = None
    return response


class TestCohereEmbeddingProviderEmbed:
    def test_single_embedding_uses_search_query_input_type(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.return_value = _make_cohere_response(
            embeddings=[[0.1, 0.2, 0.3]]
        )

        response = provider.embed("hello world")

        mock_client.embed.assert_called_once_with(
            model=MODEL,
            texts=["hello world"],
            input_type="search_query",
            embedding_types=["float"],
        )
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.model == MODEL
        assert response.usage.total_tokens == 12


class TestCohereEmbeddingProviderEmbedBatch:
    def test_batch_embedding_uses_search_document_input_type(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.return_value = _make_cohere_response(
            embeddings=[[0.1], [0.2]]
        )

        batch = provider.embed_batch(["alpha", "beta"])

        mock_client.embed.assert_called_once_with(
            model=MODEL,
            texts=["alpha", "beta"],
            input_type="search_document",
            embedding_types=["float"],
        )
        assert [vector.embedding for vector in batch.embeddings] == [[0.1], [0.2]]

    def test_large_batch_is_split_into_chunks_of_ninety(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        texts = [f"chunk-{index}" for index in range(COHERE_EMBED_BATCH_SIZE + 1)]
        mock_client.embed.side_effect = [
            _make_cohere_response(
                embeddings=[[float(index)] for index in range(COHERE_EMBED_BATCH_SIZE)],
                input_tokens=100,
            ),
            _make_cohere_response(
                embeddings=[[float(COHERE_EMBED_BATCH_SIZE)]],
                input_tokens=5,
            ),
        ]

        batch = provider.embed_batch(texts)

        assert mock_client.embed.call_count == 2
        first_call_texts = mock_client.embed.call_args_list[0].kwargs["texts"]
        second_call_texts = mock_client.embed.call_args_list[1].kwargs["texts"]
        assert len(first_call_texts) == COHERE_EMBED_BATCH_SIZE
        assert len(second_call_texts) == 1
        assert len(batch.embeddings) == COHERE_EMBED_BATCH_SIZE + 1
        assert batch.usage.prompt_tokens == 105
        assert batch.usage.total_tokens == 105

    def test_empty_input_returns_synthetic_zero_usage_without_api_call(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        batch = provider.embed_batch([])

        assert batch.embeddings == []
        assert batch.usage == EmbeddingUsage(prompt_tokens=0, total_tokens=0)
        mock_client.embed.assert_not_called()


class TestCohereEmbeddingProviderErrors:
    def test_rate_limit_error_is_converted(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.side_effect = CohereTooManyRequestsError(
            headers={"x-request-id": "req-1"},
            body={"message": "rate limited"},
        )

        with pytest.raises(EmbeddingRateLimitError):
            provider.embed_batch(["hello"])

    def test_api_error_is_converted(
        self, provider: CohereEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embed.side_effect = CohereApiError(
            headers={"x-request-id": "req-1"},
            body={"message": "server error"},
        )

        with pytest.raises(EmbeddingProviderError):
            provider.embed_batch(["hello"])
