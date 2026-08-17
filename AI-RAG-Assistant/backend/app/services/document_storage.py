import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_UPLOAD_EXTENSIONS,
    DOCUMENTS_DIR,
    DOCUMENT_CONTENT_TYPE_IMAGE,
    DOCUMENT_CONTENT_TYPE_TEXT,
    IMAGES_DIR,
)
from app.services.document_exceptions import DocumentNotFoundError


class UnsupportedDocumentType(Exception):
    """Raised when an uploaded file's extension isn't one we accept."""


def classify_upload_extension(filename: str) -> tuple[str, str]:
    """Return (extension, content_type) for an upload filename."""
    extension = Path(filename).suffix.lower()

    if extension in ALLOWED_DOCUMENT_EXTENSIONS:
        return extension, DOCUMENT_CONTENT_TYPE_TEXT
    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return extension, DOCUMENT_CONTENT_TYPE_IMAGE

    allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
    raise UnsupportedDocumentType(
        f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}."
    )


def _validate_document_id(document_id: str) -> None:
    if not document_id or "/" in document_id or "\\" in document_id or ".." in document_id:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")


def _upload_directory(content_type: str) -> Path:
    if content_type == DOCUMENT_CONTENT_TYPE_IMAGE:
        return IMAGES_DIR
    if content_type == DOCUMENT_CONTENT_TYPE_TEXT:
        return DOCUMENTS_DIR
    raise ValueError(f"Unsupported content_type '{content_type}'.")


def save_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Validate and save an uploaded file, returning (document_id, content_type)."""
    extension, content_type = classify_upload_extension(file.filename or "")

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}{extension}"
    target_dir = _upload_directory(content_type)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / stored_filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return document_id, content_type


def save_uploaded_document(file: UploadFile) -> str:
    """Backward-compatible helper for text document uploads."""
    document_id, content_type = save_uploaded_file(file)
    if content_type != DOCUMENT_CONTENT_TYPE_TEXT:
        raise UnsupportedDocumentType(
            "save_uploaded_document accepts text documents only."
        )
    return document_id


def resolve_document_path(document_id: str) -> Path:
    """Returns the on-disk path for a previously uploaded document_id.

    Uploads are stored as `{document_id}{extension}` under DOCUMENTS_DIR or
    IMAGES_DIR. We match by stem so the caller does not need to know the
    extension or content type.
    """
    _validate_document_id(document_id)

    for directory in (DOCUMENTS_DIR, IMAGES_DIR):
        matches = [
            path
            for path in directory.glob(f"{document_id}.*")
            if path.is_file() and path.stem == document_id
        ]
        if not matches:
            continue
        if len(matches) > 1:
            raise DocumentNotFoundError(
                f"Multiple files found for document_id '{document_id}'."
            )
        return matches[0]

    raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
