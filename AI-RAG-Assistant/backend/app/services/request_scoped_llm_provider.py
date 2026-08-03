from collections.abc import Generator

from llm_sdk.factories import ProviderFactory
from llm_sdk.models import LLMRequest, LLMResponse, LLMResponseChunk
from llm_sdk.providers.llm_provider import LLMProvider


class RequestScopedLLMProvider(LLMProvider):
    """Routes each LLM call to the provider named on the request."""

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        return ProviderFactory.create(request.provider).generate_response(request)

    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:
        yield from ProviderFactory.create(request.provider).generate_stream(request)
