"""Domain exceptions for the embedding pipeline."""


class EmbeddingError(Exception):
    """Base class for embedding pipeline failures."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider API call fails."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Raised when the provider rate-limits embedding requests."""
