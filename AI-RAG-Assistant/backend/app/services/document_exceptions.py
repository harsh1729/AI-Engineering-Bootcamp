"""Domain exceptions for the document pipeline (storage, parsing, later retrieval)."""


class DocumentServiceError(Exception):
    """Base class for document-pipeline failures."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when no file on disk matches the given document_id."""


class UnsupportedDocumentTypeError(DocumentServiceError):
    """Raised when the file extension has no registered parser, or the
    format is known but intentionally unsupported (e.g. legacy .doc)."""


class DocumentTooLargeError(DocumentServiceError):
    """Raised when an upload exceeds MAX_DOCUMENT_SIZE_BYTES."""


class DocumentParsingError(DocumentServiceError):
    """Raised when a registered parser fails while reading a document."""
