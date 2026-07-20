import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.models.document import DocumentUploadResponse, ParsedDocument
from app.services.document_exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from app.services.document_parser import document_parser
from app.services.document_storage import UnsupportedDocumentType, save_uploaded_document
from app.services.parsers.registry import default_parser_registry
from app.services.upload_validation import enforce_upload_size_limit, validate_upload_extension

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/documents/upload")
def upload_document(file: UploadFile) -> DocumentUploadResponse:
    try:
        # Policy checks happen before any write so oversized / rejected files
        # never land in DOCUMENTS_DIR.
        validate_upload_extension(file.filename)
        enforce_upload_size_limit(file)
        document_id = save_uploaded_document(file)
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Document upload failed.") from exc

    logger.info(
        "Document uploaded document_id=%s filename=%s",
        document_id,
        file.filename or "",
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "",
        status="uploaded",
    )


@router.get("/documents/{document_id}/parse")
def parse_document(document_id: str) -> ParsedDocument:
    """TEMPORARY: parser isolation check. Remove once RAG wiring is validated."""
    logger.info("TEMP parse request document_id=%s", document_id)

    try:
        parsed = document_parser.parse(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParsingError as exc:
        logger.exception("Document parsing failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document parsing failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected parse failure for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Document parsing failed.") from exc

    extension = Path(parsed.filename).suffix.lower()
    try:
        parser_name = type(default_parser_registry.get(extension)).__name__
    except UnsupportedDocumentTypeError:
        parser_name = "unknown"

    logger.info(
        "Parse success document_id=%s filename=%s parser=%s extracted_text_length=%d",
        parsed.document_id,
        parsed.filename,
        parser_name,
        len(parsed.extracted_text),
    )
    return parsed
