from pathlib import Path

from app.models.document import ParsedDocument
from app.services.document_exceptions import DocumentParsingError, DocumentServiceError
from app.services.document_metadata_service import get_document_metadata
from app.services.document_storage import resolve_document_path
from app.services.parsers.registry import DocumentParserRegistry, default_parser_registry


class DocumentParser:
    """Coordinates locating an uploaded file and dispatching to a format parser.

    This class contains no format-specific extraction logic - that lives in
    BaseDocumentParser strategies registered on DocumentParserRegistry.
    """

    def __init__(self, registry: DocumentParserRegistry | None = None) -> None:
        self._registry = registry or default_parser_registry

    def parse(self, document_id: str, *, original_filename: str | None = None) -> ParsedDocument:
        if original_filename is not None:
            filename = Path(original_filename).name
        else:
            filename = get_document_metadata(document_id).original_filename

        file_path = resolve_document_path(document_id)
        parser = self._registry.get(file_path.suffix.lower())

        try:
            extracted_text = parser.parse(file_path)
        except DocumentServiceError:
            # Domain errors (unsupported type, parse failure, etc.) pass through.
            raise
        except Exception as exc:
            # Wrap unexpected library failures so callers only handle our hierarchy.
            raise DocumentParsingError(
                f"Failed to parse document '{document_id}' "
                f"({filename}): {exc}"
            ) from exc

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            extracted_text=extracted_text,
        )


document_parser = DocumentParser()
