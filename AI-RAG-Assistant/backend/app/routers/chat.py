import logging
from typing import Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_sdk.factories import ProviderFactory
from llm_sdk.models import LLMMessage, LLMRequest

from app.models.chat import ChatRequest, ChatResponse
from app.services.usage_tracker import UsageLimitExceeded, usage_tracker

logger = logging.getLogger(__name__)
router = APIRouter()


def _truncation_warning(finish_reason: str | None) -> str | None:
    """Providers report token-limit cutoffs with different vocabularies
    (OpenAI: "max_output_tokens", Claude: "max_tokens", Gemini: "MAX_TOKENS").
    Checking for both "max" and "token" (rather than one fixed substring)
    covers all of them without needing a shared enum across the SDK."""
    reason = (finish_reason or "").lower()
    if "max" in reason and "token" in reason:
        return "The response was cut off because the token limit was reached."
    return None


def _build_llm_request(request: ChatRequest) -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(role=message.role, content=message.content)
            for message in request.messages
        ],
        provider=request.provider,
        model=request.model,
    )


def _enforce_usage_limit(guest_id: str) -> None:
    try:
        usage_tracker.record_interaction(guest_id)
    except UsageLimitExceeded as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _log_document_ids(request: ChatRequest) -> None:
    """Logs which documents (if any) the client attached to this request, so
    the upload -> chat wiring can be verified end-to-end before retrieval is
    implemented. This is the only thing document_ids is used for right now."""
    if request.document_ids:
        logger.info(
            "Chat request from guest_id=%s attached document_ids=%s",
            request.guest_id,
            request.document_ids,
        )


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    _enforce_usage_limit(request.guest_id)
    _log_document_ids(request)

    llm_request = _build_llm_request(request)
    provider = ProviderFactory.create(request.provider)

    try:
        llm_response = provider.generate_response(llm_request)
    except Exception as exc:
        logger.exception("LLM provider request failed")
        raise HTTPException(status_code=502, detail="LLM provider request failed.") from exc

    return ChatResponse(
        response=llm_response.text or "",
        warning=_truncation_warning(llm_response.finish_reason),
    )


def _stream_text_chunks(provider, llm_request: LLMRequest) -> Generator[str, None, None]:
    """Runs in Starlette's threadpool (StreamingResponse wraps sync generators
    automatically), so the provider's blocking API calls never block the event
    loop. Errors are yielded as a trailing text chunk rather than raised,
    since the 200 response and headers are already sent once streaming starts."""
    try:
        for chunk in provider.generate_stream(llm_request):
            if chunk.text:
                yield chunk.text
    except Exception:
        logger.exception("LLM provider streaming request failed")
        yield "\n[The assistant could not respond. Please try again.]"


@router.post("/chat/stream", response_class=StreamingResponse)
def chat_stream(request: ChatRequest) -> StreamingResponse:
    _enforce_usage_limit(request.guest_id)
    _log_document_ids(request)

    llm_request = _build_llm_request(request)
    provider = ProviderFactory.create(request.provider)

    return StreamingResponse(
        _stream_text_chunks(provider, llm_request),
        media_type="text/plain",
    )
