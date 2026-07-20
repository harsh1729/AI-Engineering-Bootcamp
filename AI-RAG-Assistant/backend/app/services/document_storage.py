import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import ALLOWED_DOCUMENT_EXTENSIONS, DOCUMENTS_DIR
from app.services.document_exceptions import DocumentNotFoundError


class UnsupportedDocumentType(Exception):
    """Raised when an uploaded file's extension isn't one we accept."""


def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        raise UnsupportedDocumentType(
            f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}."
        )

    return extension


def save_uploaded_document(file: UploadFile) -> str:
    """Validates and saves an uploaded document to disk, returning a unique
    document_id that later pipeline phases (parsing, chunking, embeddings)
    will use to look up this file. Storage is intentionally the only
    concern here - nothing about the file's contents is read or inspected.
    """
    extension = _validate_extension(file.filename or "")

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}{extension}"

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / stored_filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return document_id


def resolve_document_path(document_id: str) -> Path:
    """Returns the on-disk path for a previously uploaded document_id.

    Uploads are stored as `{document_id}{extension}` under DOCUMENTS_DIR.
    We match by stem so the caller does not need to know the extension.
    """
    if not document_id or "/" in document_id or "\\" in document_id or ".." in document_id:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    matches = [
        path
        for path in DOCUMENTS_DIR.glob(f"{document_id}.*")
        if path.is_file() and path.stem == document_id
    ]

    if not matches:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    if len(matches) > 1:
        # Should never happen with UUID filenames; fail loudly if it does.
        raise DocumentNotFoundError(
            f"Multiple files found for document_id '{document_id}'."
        )

    return matches[0]
