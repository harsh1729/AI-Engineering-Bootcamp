import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile

from app.auth.deps import get_optional_current_user
from app.database.deps import get_document_access_service, get_document_repository
from app.database.models.stored_document import Document
from app.database.models.user import User
from app.database.repositories.document_repository import DocumentRepository
from app.models.document import DocumentListResponse, DocumentSummary, DocumentUploadResponse
from app.models.rag_config import (
    DEFAULT_RAG_OPTIONS,
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    RagOptionsCatalogItem,
    RagOptionsCatalogResponse,
    PineconeIndexCatalogEntry,
    PgVectorTableCatalogEntry,
    QdrantCollectionCatalogEntry,
    VectorStoreType,
)
from app.config import (
    PGVECTOR_ENABLED,
    PINECONE_API_KEY,
    QDRANT_URL,
    pgvector_tables_catalog,
    pinecone_indexes_catalog,
    qdrant_collections_catalog,
    resolve_embedding_model,
)
from app.services.document_access_service import DocumentAccessDeniedError, DocumentAccessService
from app.services.document_exceptions import (
    DocumentIngestionError,
    DocumentNotFoundError,
    VectorDimensionMismatchError,
    DocumentParsingError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from app.services.document_metadata_service import save_indexed_document
from app.services.document_storage import UnsupportedDocumentType, save_uploaded_file
from app.services.rag_pipeline_builder import build_document_ingestion_service
from app.services.upload_validation import enforce_upload_size_limit, validate_upload_extension

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_guest_id(guest_id: str | None) -> str:
    if not guest_id:
        raise HTTPException(
            status_code=400,
            detail="guest_id is required for unauthenticated requests.",
        )
    return guest_id


def _to_document_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        id=str(document.id),
        filename=document.filename,
        content_type=document.content_type,
        chunking_strategy=document.chunking_strategy,
        embedding_provider=document.embedding_provider,
        vector_db=document.vector_db,
        chunk_count=document.chunk_count,
        user_id=str(document.user_id) if document.user_id is not None else None,
        guest_id=document.guest_id,
    )


def _parse_rag_options(raw_options: str | None) -> RagOptions:
    if not raw_options:
        return DEFAULT_RAG_OPTIONS.model_copy()

    try:
        payload = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid rag_options JSON.") from exc

    try:
        return RagOptions.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid rag_options payload.") from exc


@router.get("/rag/options")
def get_rag_options_catalog() -> RagOptionsCatalogResponse:
    return RagOptionsCatalogResponse(
        chunking_strategies=[
            RagOptionsCatalogItem(value=ChunkingStrategy.RECURSIVE, label="Recursive"),
            RagOptionsCatalogItem(value=ChunkingStrategy.CHARACTER, label="Character"),
            RagOptionsCatalogItem(value=ChunkingStrategy.SENTENCE, label="Sentence"),
            RagOptionsCatalogItem(value=ChunkingStrategy.HEADER_AWARE, label="Header Aware"),
        ],
        embedding_providers=[
            RagOptionsCatalogItem(value=EmbeddingProviderType.OPENAI, label="OpenAI"),
            RagOptionsCatalogItem(value=EmbeddingProviderType.VOYAGE, label="Voyage AI"),
            RagOptionsCatalogItem(value=EmbeddingProviderType.COHERE, label="Cohere"),
        ],
        vector_stores=[
            RagOptionsCatalogItem(value=VectorStoreType.CHROMA, label="Chroma"),
            RagOptionsCatalogItem(
                value=VectorStoreType.PINECONE,
                label="Pinecone",
                enabled=bool(PINECONE_API_KEY),
            ),
            RagOptionsCatalogItem(
                value=VectorStoreType.PGVECTOR,
                label="pgvector",
                enabled=PGVECTOR_ENABLED,
            ),
            RagOptionsCatalogItem(
                value=VectorStoreType.QDRANT,
                label="Qdrant",
                enabled=bool(QDRANT_URL),
            ),
        ],
        embedding_models={
            provider.value: resolve_embedding_model(provider.value)
            for provider in EmbeddingProviderType
        },
        defaults=DEFAULT_RAG_OPTIONS,
        pinecone_indexes=(
            {
                provider: PineconeIndexCatalogEntry(**entry)
                for provider, entry in pinecone_indexes_catalog().items()
            }
            if PINECONE_API_KEY
            else None
        ),
        pgvector_tables=(
            {
                provider: PgVectorTableCatalogEntry(**entry)
                for provider, entry in pgvector_tables_catalog().items()
            }
            if PGVECTOR_ENABLED
            else None
        ),
        qdrant_collections=(
            {
                provider: QdrantCollectionCatalogEntry(**entry)
                for provider, entry in qdrant_collections_catalog().items()
            }
            if QDRANT_URL
            else None
        ),
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    guest_id: str | None = Query(default=None, min_length=1),
    document_access: DocumentAccessService = Depends(get_document_access_service),
    current_user: User | None = Depends(get_optional_current_user),
) -> DocumentListResponse:
    if current_user is not None:
        user_id: uuid.UUID | None = current_user.id
        resolved_guest_id = guest_id
    else:
        user_id = None
        resolved_guest_id = _require_guest_id(guest_id)

    try:
        documents = await document_access.list_documents(
            user_id=user_id,
            guest_id=resolved_guest_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentListResponse(
        documents=[_to_document_summary(document) for document in documents],
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    rag_options: str | None = Form(default=None),
    guest_id: str | None = Form(default=None),
    document_repository: DocumentRepository = Depends(get_document_repository),
    current_user: User | None = Depends(get_optional_current_user),
) -> DocumentUploadResponse:
    if current_user is not None:
        owner_user_id: uuid.UUID | None = current_user.id
        owner_guest_id: str | None = None
    else:
        owner_user_id = None
        owner_guest_id = _require_guest_id(guest_id)

    filename = file.filename or ""
    original_filename = Path(filename).name
    options = _parse_rag_options(rag_options)
    selected = options.model_dump(mode="json")
    logger.info(
        "Selected RAG options for upload filename=%s form_field_provided=%s options=%s",
        filename,
        rag_options is not None,
        selected,
    )

    try:
        _, content_type = validate_upload_extension(filename)
        enforce_upload_size_limit(file, content_type=content_type)
        document_id, content_type = save_uploaded_file(file)
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Document upload failed.") from exc

    logger.info("Document saved document_id=%s filename=%s", document_id, filename)

    ingestion_service = build_document_ingestion_service(options)

    try:
        indexing_result = ingestion_service.index_document(
            document_id,
            chunking_strategy=options.chunking_strategy.value,
            original_filename=original_filename,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParsingError as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    except VectorDimensionMismatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentIngestionError as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    except Exception as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc

    try:
        await save_indexed_document(
            document_repository,
            document_id=document_id,
            filename=original_filename,
            rag_options=options,
            chunk_count=indexing_result.indexed_chunk_count,
            content_type=content_type,
            user_id=owner_user_id,
            guest_id=owner_guest_id,
        )
    except Exception as exc:
        logger.exception("Document metadata persistence failed for document_id=%s", document_id)
        raise HTTPException(
            status_code=500,
            detail="Document metadata persistence failed.",
        ) from exc

    logger.info(
        "Document processed document_id=%s filename=%s content_type=%s indexed_chunk_count=%d chunking_strategy=%s",
        document_id,
        filename,
        content_type,
        indexing_result.indexed_chunk_count,
        options.chunking_strategy.value,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=original_filename,
        status="processed",
        indexed_chunk_count=indexing_result.indexed_chunk_count,
    )
