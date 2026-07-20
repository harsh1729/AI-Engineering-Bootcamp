"""Format-specific document parsers used by DocumentParser.

Import from this package via the registry - callers outside parsers/
should not construct individual strategies directly.
"""

from app.services.parsers.base import BaseDocumentParser
from app.services.parsers.registry import DocumentParserRegistry, default_parser_registry

__all__ = [
    "BaseDocumentParser",
    "DocumentParserRegistry",
    "default_parser_registry",
]
