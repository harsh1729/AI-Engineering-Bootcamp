from typing import Generator

from google import genai
from google.genai import errors as genai_errors

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL
from models import LLMRequest, LLMResponse, LLMResponseChunk
from providers import LLMProvider
from serializers import GeminiRequestSerializer, GeminiResponseSerializer, GeminiResponseChunkSerializer


class GeminiProvider(LLMProvider):

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.request_serializer = GeminiRequestSerializer()
        self.response_serializer = GeminiResponseSerializer()
        self.response_chunk_serializer = GeminiResponseChunkSerializer()

    def generate_response(self, request: LLMRequest) -> LLMResponse:

        client_response = self._call_with_retry(
            lambda model: self.client.models.generate_content(
                **self.request_serializer.serialize(request, model=model)
            ),
            operation="Gemini API request",
        )
        return self.response_serializer.serialize(client_response)

    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:

        def stream_chunks(model: str | None) -> Generator[LLMResponseChunk, None, None]:
            payload = self.request_serializer.serialize(request, model=model)
            client_stream = self.client.models.generate_content_stream(**payload)

            for chunk in client_stream:
                llm_chunk = self.response_chunk_serializer.serialize(chunk)

                if llm_chunk:
                    yield llm_chunk

        yield from self._iter_with_retry(
            stream_chunks,
            operation="Gemini streaming API request",
        )

    def _get_retry_targets(self) -> list[str | None]:

        models: list[str | None] = [GEMINI_MODEL]

        if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
            models.append(GEMINI_FALLBACK_MODEL)

        return models

    def _is_retryable_error(self, exc: Exception) -> bool:

        return isinstance(exc, genai_errors.ServerError) and exc.code in (429, 503)
