from openai import OpenAI

from app.config import OPENAI_API_KEY, resolve_embedding_model
from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingUsage
from app.embeddings.embedding_provider_errors import raise_embedding_provider_error
from app.embeddings.embedding_provider_utils import (
    build_batch_from_indexed_items,
    empty_batch_response,
)
from app.models.rag_config import EmbeddingProviderType

# OpenAI allows up to 2,048 inputs per embeddings request; stay below for headroom.
OPENAI_EMBED_BATCH_SIZE = 1000


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or resolve_embedding_model(EmbeddingProviderType.OPENAI)

        if client is None:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured")
            client = OpenAI(api_key=OPENAI_API_KEY)

        self._client = client

    def embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        if not texts:
            return empty_batch_response()

        all_embeddings: list = []
        prompt_tokens = 0
        total_tokens = 0

        for start in range(0, len(texts), OPENAI_EMBED_BATCH_SIZE):
            batch_texts = texts[start : start + OPENAI_EMBED_BATCH_SIZE]
            batch_response = self._call_embed_batch(batch_texts)
            all_embeddings.extend(batch_response.embeddings)
            prompt_tokens += batch_response.usage.prompt_tokens
            total_tokens += batch_response.usage.total_tokens

        return EmbeddingBatchResponse(
            embeddings=all_embeddings,
            usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
        )

    def _call_embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
        except Exception as exc:
            raise_embedding_provider_error(exc)

        return self._map_response(response, expected_count=len(texts))

    def _map_response(self, response, expected_count: int) -> EmbeddingBatchResponse:
        items = ((item.index, item.embedding) for item in response.data)
        return build_batch_from_indexed_items(
            items,
            expected_count=expected_count,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )
