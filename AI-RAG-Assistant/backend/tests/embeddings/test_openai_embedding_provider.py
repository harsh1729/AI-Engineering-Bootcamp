from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from app.embeddings.embedding_exceptions import EmbeddingProviderError, EmbeddingRateLimitError
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingUsage, EmbeddingVector
from app.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider

MODEL = "text-embedding-3-small"


def _make_openai_response(
    *,
    embeddings: list[list[float]],
    model: str = MODEL,
    prompt_tokens: int = 8,
    total_tokens: int = 8,
    indices: list[int] | None = None,
) -> MagicMock:
    if indices is None:
        indices = list(range(len(embeddings)))

    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.total_tokens = total_tokens
    response.data = [
        MagicMock(embedding=embedding, index=index)
        for embedding, index in zip(embeddings, indices, strict=True)
    ]
    return response


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def provider(mock_client: MagicMock) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(client=mock_client, model=MODEL)


class TestOpenAIEmbeddingProviderEmbed:
    def test_single_embedding_delegates_to_embed_batch(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1, 0.2, 0.3]]
        )

        response = provider.embed("hello world")

        mock_client.embeddings.create.assert_called_once_with(
            input=["hello world"],
            model=MODEL,
        )
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.model == MODEL
        assert response.dimensions == 3
        assert response.usage.prompt_tokens == 8
        assert response.usage.total_tokens == 8

    def test_embed_returns_first_batch_result(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1, 0.2]]
        )

        response = provider.embed("hello")

        assert response.embedding == [0.1, 0.2]

    def test_embed_calls_embed_batch_with_single_item(
        self, provider: OpenAIEmbeddingProvider
    ) -> None:
        batch = EmbeddingBatchResponse(
            embeddings=[EmbeddingVector(embedding=[0.1, 0.2], model=MODEL)],
            usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3),
        )

        with patch.object(
            provider,
            "embed_batch",
            return_value=batch,
        ) as mock_embed_batch:
            response = provider.embed("hello")

        mock_embed_batch.assert_called_once_with(["hello"])
        assert response.embedding == [0.1, 0.2]
        assert response.model == MODEL
        assert response.usage == batch.usage


class TestOpenAIEmbeddingProviderEmbedBatch:
    def test_batch_embedding_makes_one_api_request(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1], [0.2], [0.3]]
        )

        batch = provider.embed_batch(["alpha", "beta", "gamma"])

        mock_client.embeddings.create.assert_called_once_with(
            input=["alpha", "beta", "gamma"],
            model=MODEL,
        )
        assert len(batch.embeddings) == 3

    def test_batch_embedding_preserves_input_order(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.model = MODEL
        response.usage.prompt_tokens = 8
        response.usage.total_tokens = 8
        response.data = [
            MagicMock(embedding=[2.0], index=1),
            MagicMock(embedding=[3.0], index=2),
            MagicMock(embedding=[1.0], index=0),
        ]
        mock_client.embeddings.create.return_value = response

        batch = provider.embed_batch(["first", "second", "third"])

        assert [vector.embedding for vector in batch.embeddings] == [[1.0], [2.0], [3.0]]

    def test_batch_embedding_maps_response_fields(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.5, 0.6], [0.7, 0.8]],
            model="text-embedding-3-large",
            prompt_tokens=12,
            total_tokens=15,
        )

        batch = provider.embed_batch(["one", "two"])
        first, second = batch.embeddings

        assert first.embedding == [0.5, 0.6]
        assert second.embedding == [0.7, 0.8]
        assert first.model == "text-embedding-3-large"
        assert second.model == "text-embedding-3-large"

    def test_batch_embedding_maps_usage_once(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1], [0.2]],
            prompt_tokens=20,
            total_tokens=25,
        )

        batch = provider.embed_batch(["one", "two"])

        assert batch.usage.prompt_tokens == 20
        assert batch.usage.total_tokens == 25
        assert all(isinstance(vector, EmbeddingVector) for vector in batch.embeddings)

    def test_empty_input_returns_synthetic_zero_usage_without_api_call(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        batch = provider.embed_batch([])

        assert batch.embeddings == []
        assert batch.usage.prompt_tokens == 0
        assert batch.usage.total_tokens == 0
        mock_client.embeddings.create.assert_not_called()


class TestOpenAIEmbeddingProviderIndexValidation:
    def test_out_of_order_api_response_returns_results_in_input_order(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.model = MODEL
        response.usage.prompt_tokens = 8
        response.usage.total_tokens = 8
        response.data = [
            MagicMock(embedding=[3.0], index=2),
            MagicMock(embedding=[1.0], index=0),
            MagicMock(embedding=[2.0], index=1),
        ]
        mock_client.embeddings.create.return_value = response

        batch = provider.embed_batch(["first", "second", "third"])

        assert [vector.embedding for vector in batch.embeddings] == [[1.0], [2.0], [3.0]]

    def test_duplicate_indices_raise_embedding_provider_error(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1], [0.2]],
            indices=[0, 0],
        )

        with pytest.raises(EmbeddingProviderError, match="Duplicate embedding response index 0"):
            provider.embed_batch(["one", "two"])

    def test_missing_indices_raise_embedding_provider_error(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1], [0.2]],
            indices=[0, 1],
        )

        with pytest.raises(
            EmbeddingProviderError,
            match=r"Missing embedding response indices: \[2\]",
        ):
            provider.embed_batch(["one", "two", "three"])

    def test_out_of_range_indices_raise_embedding_provider_error(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1], [0.2]],
            indices=[0, 2],
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="Embedding response index 2 is out of range for batch size 2",
        ):
            provider.embed_batch(["one", "two"])

    def test_negative_indices_raise_embedding_provider_error(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.return_value = _make_openai_response(
            embeddings=[[0.1]],
            indices=[-1],
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="Embedding response index -1 is out of range for batch size 1",
        ):
            provider.embed_batch(["one"])


class TestOpenAIEmbeddingProviderErrors:
    def test_api_status_error_is_converted(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(500, request=request)
        mock_client.embeddings.create.side_effect = APIStatusError(
            "server error",
            response=response,
            body=None,
        )

        with pytest.raises(EmbeddingProviderError, match="server error"):
            provider.embed_batch(["hello"])

    def test_rate_limit_error_is_converted(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(429, request=request)
        mock_client.embeddings.create.side_effect = RateLimitError(
            "rate limited",
            response=response,
            body=None,
        )

        with pytest.raises(EmbeddingRateLimitError, match="rate limited"):
            provider.embed_batch(["hello"])

    def test_unexpected_exception_propagates(
        self, provider: OpenAIEmbeddingProvider, mock_client: MagicMock
    ) -> None:
        mock_client.embeddings.create.side_effect = RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            provider.embed_batch(["hello"])
