"""Upload-time validation helpers used by the documents router.

Kept separate from document_storage so storage remains responsible only for
persisting bytes, while the endpoint owns request-level policy (extension
order + size limit) before any file is written.
"""

from pathlib import Path

from fastapi import UploadFile

from app.config import (
    DOCUMENT_CONTENT_TYPE_IMAGE,
    MAX_DOCUMENT_SIZE_BYTES,
    MAX_DOCUMENT_SIZE_MB,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
)
from app.services.document_exceptions import DocumentTooLargeError
from app.services.document_storage import UnsupportedDocumentType, classify_upload_extension


def validate_upload_extension(filename: str | None) -> tuple[str, str]:
    """Raises UnsupportedDocumentType when the extension is not allowed.

    Returns (extension, content_type).
    """
    try:
        return classify_upload_extension(filename or "")
    except UnsupportedDocumentType:
        raise


def enforce_upload_size_limit(file: UploadFile, *, content_type: str) -> None:
    """Rejects uploads larger than the content-type size limit.

    Starlette buffers multipart parts in a seekable SpooledTemporaryFile, so
    we measure size via seek/tell and rewind before the caller saves.
    """
    stream = file.file
    current = stream.tell()
    stream.seek(0, 2)  # end
    size = stream.tell()
    stream.seek(current)

    if content_type == DOCUMENT_CONTENT_TYPE_IMAGE:
        max_bytes = MAX_IMAGE_SIZE_BYTES
        max_mb = MAX_IMAGE_SIZE_MB
    else:
        max_bytes = MAX_DOCUMENT_SIZE_BYTES
        max_mb = MAX_DOCUMENT_SIZE_MB

    if size > max_bytes:
        label = "image" if content_type == DOCUMENT_CONTENT_TYPE_IMAGE else "document"
        raise DocumentTooLargeError(
            f"Maximum {label} size is {max_mb} MB."
        )
