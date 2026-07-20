"""Upload-time validation helpers used by the documents router.

Kept separate from document_storage so storage remains responsible only for
persisting bytes, while the endpoint owns request-level policy (extension
order + size limit) before any file is written.
"""

from pathlib import Path

from fastapi import UploadFile

from app.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    MAX_DOCUMENT_SIZE_BYTES,
    MAX_DOCUMENT_SIZE_MB,
)
from app.services.document_exceptions import DocumentTooLargeError
from app.services.document_storage import UnsupportedDocumentType


def validate_upload_extension(filename: str | None) -> str:
    """Raises UnsupportedDocumentType when the extension is not allowed."""
    extension = Path(filename or "").suffix.lower()

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        raise UnsupportedDocumentType(
            f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}."
        )

    return extension


def enforce_upload_size_limit(file: UploadFile) -> None:
    """Rejects uploads larger than MAX_DOCUMENT_SIZE_BYTES without buffering
    the whole body into memory.

    Starlette buffers multipart parts in a seekable SpooledTemporaryFile, so
    we measure size via seek/tell and rewind before the caller saves.
    """
    stream = file.file
    current = stream.tell()
    stream.seek(0, 2)  # end
    size = stream.tell()
    stream.seek(current)

    if size > MAX_DOCUMENT_SIZE_BYTES:
        raise DocumentTooLargeError(
            f"Maximum document size is {MAX_DOCUMENT_SIZE_MB} MB."
        )
