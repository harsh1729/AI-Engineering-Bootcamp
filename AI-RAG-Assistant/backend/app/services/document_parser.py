from app.models.document import ParsedDocument
from app.services.document_exceptions import DocumentParsingError, DocumentServiceError
from app.services.document_storage import get_document_metadata, resolve_document_path
from app.services.parsers.registry import DocumentParserRegistry, default_parser_registry


class DocumentParser:
    """Coordinates locating an uploaded file and dispatching to a format parser.

    This class contains no format-specific extraction logic - that lives in
    BaseDocumentParser strategies registered on DocumentParserRegistry.
    """

    def __init__(self, registry: DocumentParserRegistry | None = None) -> None:
        self._registry = registry or default_parser_registry

    def parse(self, document_id: str) -> ParsedDocument:
        metadata = get_document_metadata(document_id)
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
                f"({metadata.original_filename}): {exc}"
            ) from exc

        return ParsedDocument(
            document_id=document_id,
            filename=metadata.original_filename,
            extracted_text=extracted_text,
        )


document_parser = DocumentParser()
