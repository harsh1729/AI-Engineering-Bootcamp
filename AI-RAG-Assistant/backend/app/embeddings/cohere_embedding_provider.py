from cohere import ClientV2 as CohereClient

from app.config import COHERE_API_KEY, resolve_embedding_model
from app.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.embeddings.embedding_models import EmbeddingBatchResponse, EmbeddingResponse
from app.embeddings.embedding_provider_errors import raise_embedding_provider_error
from app.embeddings.embedding_provider_utils import (
    build_batch_from_ordered_vectors,
    build_embedding_response,
    empty_batch_response,
)
from app.models.rag_config import EmbeddingProviderType

# Cohere allows up to 96 texts per embed request; stay below that for headroom.
COHERE_EMBED_BATCH_SIZE = 90


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the Cohere Embed API."""

    def __init__(
        self,
        client: CohereClient | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or resolve_embedding_model(EmbeddingProviderType.COHERE)

        if client is None:
            if not COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY is not configured")
            client = CohereClient(api_key=COHERE_API_KEY)

        self._client = client

    def embed(self, text: str) -> EmbeddingResponse:
        return build_embedding_response(
            self._embed_texts([text], input_type="search_query")
        )

    def embed_batch(self, texts: list[str]) -> EmbeddingBatchResponse:
        return self._embed_texts(texts, input_type="search_document")

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

        for start in range(0, len(texts), COHERE_EMBED_BATCH_SIZE):
            batch_texts = texts[start : start + COHERE_EMBED_BATCH_SIZE]
            batch_embeddings, batch_prompt_tokens, batch_total_tokens = self._call_embed(
                batch_texts,
                input_type=input_type,
            )
            all_embeddings.extend(batch_embeddings)
            prompt_tokens += batch_prompt_tokens
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
    ) -> tuple[list[list[float]], int, int]:
        try:
            response = self._client.embed(
                model=self._model,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
            )
        except Exception as exc:
            raise_embedding_provider_error(exc)

        float_embeddings = response.embeddings.float or []
        total_tokens = self._extract_total_tokens(response)
        return float_embeddings, total_tokens, total_tokens

    def _extract_total_tokens(self, response) -> int:
        if response.meta and response.meta.billed_units:
            billed = response.meta.billed_units.input_tokens
            if billed is not None:
                return int(billed)

        if response.meta and response.meta.tokens and response.meta.tokens.input_tokens:
            return int(response.meta.tokens.input_tokens)

        return 0
