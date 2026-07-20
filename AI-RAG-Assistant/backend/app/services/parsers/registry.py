from app.services.document_exceptions import UnsupportedDocumentTypeError
from app.services.parsers.base import BaseDocumentParser
from app.services.parsers.excel_parser import XlsxParser
from app.services.parsers.legacy_parser import UnsupportedLegacyParser
from app.services.parsers.pdf_parser import PdfParser
from app.services.parsers.powerpoint_parser import PptxParser
from app.services.parsers.rtf_parser import RtfParser
from app.services.parsers.txt_parser import TxtParser
from app.services.parsers.word_parser import DocxParser


class DocumentParserRegistry:
    """Maps file extensions to BaseDocumentParser strategies.

    Adding a new format means registering one more entry - DocumentParser
    itself never grows an if/elif chain over file types.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, BaseDocumentParser] = {
            ".txt": TxtParser(),
            ".pdf": PdfParser(),
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(),
            ".pptx": PptxParser(),
            ".rtf": RtfParser(),
            # Legacy binary Office formats: registered so callers get a clear
            # UnsupportedDocumentTypeError instead of a generic "unknown type".
            ".doc": UnsupportedLegacyParser(".doc", "Microsoft Word 97-2003"),
            ".xls": UnsupportedLegacyParser(".xls", "Microsoft Excel 97-2003"),
            ".ppt": UnsupportedLegacyParser(".ppt", "Microsoft PowerPoint 97-2003"),
        }

    def get(self, extension: str) -> BaseDocumentParser:
        normalized = extension.lower()
        parser = self._parsers.get(normalized)
        if parser is None:
            supported = ", ".join(sorted(self._parsers))
            raise UnsupportedDocumentTypeError(
                f"Unsupported file type '{normalized or 'unknown'}'. "
                f"Supported types: {supported}."
            )
        return parser

    def register(self, extension: str, parser: BaseDocumentParser) -> None:
        """Allows tests or future plugins to swap/extend parsers without
        editing this module."""
        self._parsers[extension.lower()] = parser


default_parser_registry = DocumentParserRegistry()
