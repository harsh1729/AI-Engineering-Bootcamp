import json
import logging

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.models.document import DocumentUploadResponse
from app.models.rag_config import (
    DEFAULT_RAG_OPTIONS,
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    RagOptionsCatalogItem,
    RagOptionsCatalogResponse,
    VectorStoreType,
)
from app.services.document_exceptions import (
    DocumentIngestionError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from app.services.document_storage import UnsupportedDocumentType, save_uploaded_document
from app.services.rag_pipeline_builder import build_document_ingestion_service
from app.services.upload_validation import enforce_upload_size_limit, validate_upload_extension

logger = logging.getLogger(__name__)
router = APIRouter()


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
        ],
        embedding_providers=[
            RagOptionsCatalogItem(value=EmbeddingProviderType.OPENAI, label="OpenAI"),
        ],
        vector_stores=[
            RagOptionsCatalogItem(value=VectorStoreType.CHROMA, label="Chroma"),
        ],
        defaults=DEFAULT_RAG_OPTIONS,
    )


@router.post("/documents/upload")
def upload_document(
    file: UploadFile,
    rag_options: str | None = Form(default=None),
) -> DocumentUploadResponse:
    filename = file.filename or ""
    options = _parse_rag_options(rag_options)
    logger.info(
        "Uploading document filename=%s rag_options=%s",
        filename,
        options.model_dump(),
    )

    try:
        validate_upload_extension(filename)
        enforce_upload_size_limit(file)
        document_id = save_uploaded_document(file)
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
        indexing_result = ingestion_service.index_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParsingError as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    except DocumentIngestionError as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    except Exception as exc:
        logger.exception("Document ingestion failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc

    logger.info(
        "Document processed document_id=%s filename=%s indexed_chunk_count=%d chunking_strategy=%s",
        document_id,
        filename,
        indexing_result.indexed_chunk_count,
        options.chunking_strategy.value,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status="processed",
        indexed_chunk_count=indexing_result.indexed_chunk_count,
    )
