from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY
from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_exceptions import EmbeddingProviderError, EmbeddingRateLimitError
from app.embeddings.embedding_models import EmbeddingResponse, EmbeddingUsage


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or EMBEDDING_MODEL

        if client is None:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured")
            client = OpenAI(api_key=OPENAI_API_KEY)

        self._client = client

    def embed(self, text: str) -> EmbeddingResponse:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        if not texts:
            return []

        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
        except RateLimitError as exc:
            raise EmbeddingRateLimitError(str(exc)) from exc
        except (APIConnectionError, APIStatusError, InternalServerError) as exc:
            raise EmbeddingProviderError(str(exc)) from exc

        return self._map_response(response, expected_count=len(texts))

    def _map_response(self, response, expected_count: int) -> list[EmbeddingResponse]:
        usage = EmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )
        results: list[EmbeddingResponse | None] = [None] * expected_count

        for item in response.data:
            index = item.index
            if index < 0 or index >= expected_count:
                raise EmbeddingProviderError(
                    f"Embedding response index {index} is out of range for batch size "
                    f"{expected_count}"
                )
            if results[index] is not None:
                raise EmbeddingProviderError(
                    f"Duplicate embedding response index {index}"
                )
            results[index] = self._map_item(item, response, usage)

        missing_indices = [index for index, result in enumerate(results) if result is None]
        if missing_indices:
            raise EmbeddingProviderError(
                f"Missing embedding response indices: {missing_indices}"
            )

        return results

    def _map_item(
        self,
        item,
        response,
        usage: EmbeddingUsage,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=item.embedding,
            model=response.model,
            usage=usage,
        )
