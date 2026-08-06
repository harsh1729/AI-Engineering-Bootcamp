"""Map vendor-specific embedding exceptions to domain errors."""

from cohere import TooManyRequestsError as CohereTooManyRequestsError
from cohere.core.api_error import ApiError as CohereApiError
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIStatusError as OpenAIAPIStatusError,
    InternalServerError as OpenAIInternalServerError,
    RateLimitError as OpenAIRateLimitError,
)
from voyageai.error import (
    APIConnectionError as VoyageAPIConnectionError,
    RateLimitError as VoyageRateLimitError,
    ServiceUnavailableError as VoyageServiceUnavailableError,
    VoyageError,
)

from app.embeddings.embedding_exceptions import EmbeddingProviderError, EmbeddingRateLimitError


def raise_embedding_provider_error(exc: Exception) -> None:
    """Convert a supported vendor exception into a domain embedding error."""
    if isinstance(exc, (OpenAIRateLimitError, VoyageRateLimitError, CohereTooManyRequestsError)):
        raise EmbeddingRateLimitError(str(exc)) from exc

    if isinstance(
        exc,
        (
            OpenAIAPIConnectionError,
            OpenAIAPIStatusError,
            OpenAIInternalServerError,
            VoyageAPIConnectionError,
            VoyageServiceUnavailableError,
            VoyageError,
            CohereApiError,
        ),
    ):
        raise EmbeddingProviderError(str(exc)) from exc

    raise exc
