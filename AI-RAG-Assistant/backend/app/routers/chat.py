import json
import logging
import re
from collections.abc import AsyncGenerator, Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from llm_sdk.factories import ProviderFactory
from llm_sdk.models import LLMRequest

from app.auth.deps import get_optional_current_user
from app.database.deps import (
    get_chat_persistence_service,
    get_document_access_service,
    get_rag_options_resolver_service,
    get_usage_tracker_service,
)
from app.database.models.user import User
from app.dependencies import get_chat_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_persistence_exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
)
from app.services.chat_persistence_service import (
    ChatPersistenceService,
    extract_latest_user_message,
)
from app.services.chat_service import ChatService, PreparedStreamRequest
from app.services.document_access_service import DocumentAccessDeniedError, DocumentAccessService
from app.services.rag_options_resolver import (
    InvalidDocumentRagConfigError,
    MixedRagConfigError,
    RagOptionsResolverService,
)
from app.services.usage_exceptions import UsageLimitExceeded
from app.services.usage_tracker import UsageTrackerService

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_guest_id(guest_id: str | None) -> str:
    if not guest_id:
        raise HTTPException(
            status_code=400,
            detail="guest_id is required for unauthenticated requests.",
        )
    return guest_id


async def _enforce_usage_limit(
    guest_id: str | None,
    current_user: User | None,
    usage_tracker: UsageTrackerService,
) -> None:
    try:
        if current_user is not None:
            await usage_tracker.record_user_interaction(current_user.id)
        else:
            await usage_tracker.record_guest_interaction(_require_guest_id(guest_id))
    except UsageLimitExceeded as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _log_document_ids(request: ChatRequest) -> None:
    if request.document_ids:
        logger.info(
            "Chat request from guest_id=%s attached document_ids=%s",
            request.guest_id,
            request.document_ids,
        )


def _provider_name(request: ChatRequest) -> str | None:
    return request.provider.value if request.provider is not None else None


async def _validate_document_ids(
    request: ChatRequest,
    *,
    current_user: User | None,
    document_access: DocumentAccessService,
) -> None:
    if not request.document_ids:
        return

    if current_user is not None:
        user_id = current_user.id
        guest_id = request.guest_id
    else:
        user_id = None
        guest_id = _require_guest_id(request.guest_id)

    try:
        await document_access.validate_document_ids(
            request.document_ids,
            user_id=user_id,
            guest_id=guest_id,
        )
    except DocumentAccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


async def _resolve_rag_options(
    request: ChatRequest,
    rag_options_resolver: RagOptionsResolverService,
) -> None:
    if not request.document_ids:
        return

    try:
        resolved = await rag_options_resolver.resolve_rag_options_for_documents(
            request.document_ids,
        )
    except MixedRagConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidDocumentRagConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.rag_options is not None and request.rag_options != resolved:
        logger.info(
            "Ignoring client rag_options=%s; using stored document config=%s",
            request.rag_options.model_dump(),
            resolved.model_dump(),
        )

    request.rag_options = resolved


async def _resolve_chat(
    persistence: ChatPersistenceService,
    request: ChatRequest,
    *,
    user_content: str,
    current_user: User | None,
) -> object:
    provider = _provider_name(request)
    if provider is None:
        raise HTTPException(status_code=400, detail="provider is required.")

    if current_user is not None:
        user_id = current_user.id
        guest_id = request.guest_id
    else:
        user_id = None
        guest_id = _require_guest_id(request.guest_id)

    try:
        return await persistence.get_or_create_chat(
            guest_id=guest_id,
            user_id=user_id,
            provider=provider,
            chat_id=request.chat_id,
            first_message=user_content if request.chat_id is None else None,
        )
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatAccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    persistence: ChatPersistenceService = Depends(get_chat_persistence_service),
    usage_tracker: UsageTrackerService = Depends(get_usage_tracker_service),
    document_access: DocumentAccessService = Depends(get_document_access_service),
    rag_options_resolver: RagOptionsResolverService = Depends(get_rag_options_resolver_service),
    current_user: User | None = Depends(get_optional_current_user),
) -> ChatResponse:
    await _enforce_usage_limit(request.guest_id, current_user, usage_tracker)
    _log_document_ids(request)

    try:
        user_content = extract_latest_user_message(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await _validate_document_ids(
        request,
        current_user=current_user,
        document_access=document_access,
    )
    await _resolve_rag_options(request, rag_options_resolver)

    chat = await _resolve_chat(
        persistence,
        request,
        user_content=user_content,
        current_user=current_user,
    )
    await persistence.append_user_message(chat, user_content)

    try:
        result = chat_service.chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail="LLM provider request failed.") from exc

    await persistence.append_assistant_message(
        chat,
        result.response,
        provider=_provider_name(request),
        model=request.model,
    )

    return ChatResponse(
        chat_id=chat.id,
        response=result.response,
        warning=result.warning,
        sources=result.sources,
    )


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


def _stream_text_chunks_from_prepared(
    prepared: PreparedStreamRequest,
    provider,
) -> Generator[str, None, None]:
    if prepared.rag_service is not None:
        try:
            yield from prepared.rag_service.stream_answer(prepared.llm_request)
        except Exception as exc:
            logger.exception("LLM provider streaming RAG request failed")
            yield f"\n{_streaming_error_message(exc)}"
        return

    yield from _stream_text_chunks(provider, prepared.llm_request)


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    persistence: ChatPersistenceService = Depends(get_chat_persistence_service),
    usage_tracker: UsageTrackerService = Depends(get_usage_tracker_service),
    document_access: DocumentAccessService = Depends(get_document_access_service),
    rag_options_resolver: RagOptionsResolverService = Depends(get_rag_options_resolver_service),
    current_user: User | None = Depends(get_optional_current_user),
) -> StreamingResponse:
    await _enforce_usage_limit(request.guest_id, current_user, usage_tracker)
    _log_document_ids(request)

    try:
        user_content = extract_latest_user_message(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await _validate_document_ids(
        request,
        current_user=current_user,
        document_access=document_access,
    )
    await _resolve_rag_options(request, rag_options_resolver)

    chat = await _resolve_chat(
        persistence,
        request,
        user_content=user_content,
        current_user=current_user,
    )
    await persistence.append_user_message(chat, user_content)

    try:
        prepared = chat_service.prepare_stream(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    provider = ProviderFactory.create(request.provider)
    response_headers = {"X-Chat-Id": str(chat.id)}
    if prepared.sources:
        response_headers["X-RAG-Sources"] = json.dumps(
            [source.model_dump() for source in prepared.sources],
        )

    async def stream_and_persist() -> AsyncGenerator[str, None]:
        accumulated: list[str] = []
        for chunk in _stream_text_chunks_from_prepared(prepared, provider):
            accumulated.append(chunk)
            yield chunk

        await persistence.append_assistant_message(
            chat,
            "".join(accumulated),
            provider=_provider_name(request),
            model=request.model,
        )

    return StreamingResponse(
        stream_and_persist(),
        media_type="text/plain",
        headers=response_headers,
    )
