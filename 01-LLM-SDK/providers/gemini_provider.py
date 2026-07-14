from dataclasses import dataclass
from typing import Any, Generator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL
from models import LLMRequest, LLMResponse, LLMResponseChunk
from models.tools import LLMToolCall, LLMToolExecutionResult
from providers import LLMProvider
from serializers import GeminiRequestSerializer, GeminiResponseSerializer, GeminiResponseChunkSerializer

MAX_TOOL_CALL_DEPTH = 10


@dataclass
class _GeminiToolFollowUp:
    assistant_content: types.Content
    tool_results: list[LLMToolExecutionResult]


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

        llm_response = self.response_serializer.serialize(client_response)

        return self._finalize_response(request, llm_response)

    def _finalize_response(
        self,
        request: LLMRequest,
        response: LLMResponse,
    ) -> LLMResponse:

        depth = 0

        while response.tool_calls:
            if depth >= MAX_TOOL_CALL_DEPTH:
                raise RuntimeError(
                    f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
                )

            tool_results = self._execute_tools(response.tool_calls)

            assistant_content = response.raw_response.candidates[0].content

            response = self.response_serializer.serialize(
                self._call_with_retry(
                    lambda model: self.client.models.generate_content(
                        **self.request_serializer.serialize_tool_results(
                            request=request,
                            assistant_content=assistant_content,
                            tool_results=tool_results,
                            model=model,
                        )
                    ),
                    operation="Gemini API follow-up request",
                )
            )

            depth += 1

        return response

    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:

        yield from self._finalize_stream(request)

    def _finalize_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:

        tool_follow_up: _GeminiToolFollowUp | None = None

        for _ in range(MAX_TOOL_CALL_DEPTH):
            stream_state = {
                "tool_calls": [],
                "assistant_parts": [],
                "seen_call_ids": set(),
            }

            def stream_turn(model: str | None) -> Generator[LLMResponseChunk, None, None]:
                if tool_follow_up is None:
                    payload = self.request_serializer.serialize(request, model=model)
                else:
                    payload = self.request_serializer.serialize_tool_results(
                        request=request,
                        assistant_content=tool_follow_up.assistant_content,
                        tool_results=tool_follow_up.tool_results,
                        model=model,
                    )

                client_stream = self.client.models.generate_content_stream(**payload)

                for chunk in client_stream:
                    self._collect_stream_tool_state(chunk, stream_state)

                    llm_chunk = self.response_chunk_serializer.serialize(chunk)

                    if not llm_chunk:
                        continue

                    if llm_chunk.is_finished and stream_state["tool_calls"]:
                        continue

                    yield llm_chunk

            operation = (
                "Gemini streaming API follow-up request"
                if tool_follow_up is not None
                else "Gemini streaming API request"
            )

            yield from self._stream_with_retry(
                stream_turn,
                operation=operation,
            )

            tool_calls: list[LLMToolCall] = stream_state["tool_calls"]

            if not tool_calls:
                return

            assistant_parts: list[types.Part] = stream_state["assistant_parts"]

            tool_follow_up = _GeminiToolFollowUp(
                assistant_content=types.Content(role="model", parts=assistant_parts),
                tool_results=self._execute_tools(tool_calls),
            )

        raise RuntimeError(
            f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
        )

    def _collect_stream_tool_state(
        self,
        chunk: types.GenerateContentResponse,
        stream_state: dict[str, Any],
    ) -> None:

        if not chunk.candidates:
            return

        candidate = chunk.candidates[0]

        if not candidate.content or not candidate.content.parts:
            return

        for part in candidate.content.parts:
            if not part.function_call:
                continue

            if part.function_call.id is None:
                raise ValueError(
                    f"Gemini function_call for '{part.function_call.name}' is missing id."
                )

            call_id = part.function_call.id

            if call_id not in stream_state["seen_call_ids"]:
                stream_state["seen_call_ids"].add(call_id)
                stream_state["tool_calls"].append(
                    LLMToolCall(
                        call_id=call_id,
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args),
                    )
                )

            stream_state["assistant_parts"].append(part)

    def _get_retry_targets(self) -> list[str | None]:

        models: list[str | None] = [GEMINI_MODEL]

        if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
            models.append(GEMINI_FALLBACK_MODEL)

        return models

    def _is_retryable_error(self, exc: Exception) -> bool:

        return isinstance(exc, genai_errors.ServerError) and exc.code in (429, 503)
