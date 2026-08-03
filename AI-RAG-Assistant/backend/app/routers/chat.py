import logging
import re
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from llm_sdk.factories import ProviderFactory
from llm_sdk.models import LLMRequest

from app.dependencies import get_chat_service
from app.llm.llm_request_builder import build_chat_llm_request
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.usage_tracker import UsageLimitExceeded, usage_tracker

logger = logging.getLogger(__name__)
router = APIRouter()


def _enforce_usage_limit(guest_id: str) -> None:
    try:
        usage_tracker.record_interaction(guest_id)
    except UsageLimitExceeded as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _log_document_ids(request: ChatRequest) -> None:
    if request.document_ids:
        logger.info(
            "Chat request from guest_id=%s attached document_ids=%s",
            request.guest_id,
            request.document_ids,
        )


@router.post("/chat")
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    _enforce_usage_limit(request.guest_id)

    try:
        return chat_service.chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail="LLM provider request failed.") from exc


def _extract_provider_error_message(exc: Exception) -> str | None:
    """Pull a user-facing message from provider SDK exceptions when available."""
    response_json = getattr(exc, "response_json", None)
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

    match = re.search(r"'message': '((?:\\'|[^'])*)'", str(exc))
    if match:
        return match.group(1).replace("\\'", "'")

    match = re.search(r'"message": "((?:\\"|[^"])*)"', str(exc))
    if match:
        return match.group(1).replace('\\"', '"')

    return None


def _streaming_error_message(exc: Exception) -> str:
    provider_message = _extract_provider_error_message(exc)
    if provider_message:
        return provider_message
    return "The assistant could not respond. Please try again."


def _stream_text_chunks(provider, llm_request: LLMRequest) -> Generator[str, None, None]:
    """Runs in Starlette's threadpool (StreamingResponse wraps sync generators
    automatically), so the provider's blocking API calls never block the event
    loop. Errors are yielded as a trailing text chunk rather than raised,
    since the 200 response and headers are already sent once streaming starts."""
    try:
        for chunk in provider.generate_stream(llm_request):
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        logger.exception("LLM provider streaming request failed")
        yield f"\n{_streaming_error_message(exc)}"


@router.post("/chat/stream", response_class=StreamingResponse)
def chat_stream(request: ChatRequest) -> StreamingResponse:
    _enforce_usage_limit(request.guest_id)
    _log_document_ids(request)

    llm_request = build_chat_llm_request(request, include_tools=True)
    provider = ProviderFactory.create(request.provider)

    return StreamingResponse(
        _stream_text_chunks(provider, llm_request),
        media_type="text/plain",
    )
