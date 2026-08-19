from voyageai import Client as VoyageClient

from app.config import VOYAGE_API_KEY, resolve_embedding_model
from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingResponse
from app.embeddings.embedding_provider_errors import raise_embedding_provider_error
from app.embeddings.embedding_provider_utils import (
    build_batch_from_ordered_vectors,
    build_embedding_response,
    empty_batch_response,
)
from app.models.rag_config import EmbeddingProviderType

# Voyage allows up to 1,000 texts per embed request; stay below for headroom.
VOYAGE_EMBED_BATCH_SIZE = 500


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the Voyage AI Embeddings API."""

    def __init__(
        self,
        client: VoyageClient | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or resolve_embedding_model(EmbeddingProviderType.VOYAGE)

        if client is None:
            if not VOYAGE_API_KEY:
                raise ValueError("VOYAGE_API_KEY is not configured")
            client = VoyageClient(api_key=VOYAGE_API_KEY)

        self._client = client

    def embed(self, text: str) -> EmbeddingResponse:
        return build_embedding_response(self._embed_texts([text], input_type="query"))

    def embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        return self._embed_texts(texts, input_type="document")

    def _embed_texts(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> EmbeddingBatchResponse:
        if not texts:
            return empty_batch_response()

        all_embeddings: list[list[float]] = []
        prompt_tokens = 0
        total_tokens = 0

        for start in range(0, len(texts), VOYAGE_EMBED_BATCH_SIZE):
            batch_texts = texts[start : start + VOYAGE_EMBED_BATCH_SIZE]
            batch_embeddings, batch_total_tokens = self._call_embed(
                batch_texts,
                input_type=input_type,
            )
            all_embeddings.extend(batch_embeddings)
            prompt_tokens += batch_total_tokens
            total_tokens += batch_total_tokens

        return build_batch_from_ordered_vectors(
            all_embeddings,
            model=self._model,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

    def _call_embed(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> tuple[list[list[float]], int]:
        try:
            response = self._client.embed(
                texts,
                model=self._model,
                input_type=input_type,
            )
        except Exception as exc:
            raise_embedding_provider_error(exc)

        return response.embeddings, response.total_tokens
