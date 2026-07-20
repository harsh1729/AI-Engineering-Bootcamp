from pathlib import Path

from app.services.document_exceptions import UnsupportedDocumentTypeError
from app.services.parsers.base import BaseDocumentParser


class UnsupportedLegacyParser(BaseDocumentParser):
    """Placeholder strategy for legacy binary Office formats (.doc, .xls, .ppt).

    Reliable text extraction for these formats needs specialized converters
    (e.g. LibreOffice) that we do not want as a hard dependency. Registering
    this parser keeps the registry complete and returns a clear error instead
    of a silent failure or a giant if/elif in DocumentParser.
    """

    def __init__(self, extension: str, format_name: str) -> None:
        self._extension = extension
        self._format_name = format_name

    def parse(self, file_path: Path) -> str:
        raise UnsupportedDocumentTypeError(
            f"Legacy format '{self._extension}' ({self._format_name}) is not "
            f"supported yet. Convert '{file_path.name}' to the modern Office "
            f"format and re-upload."
        )
