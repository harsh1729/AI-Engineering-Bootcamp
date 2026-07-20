from pathlib import Path

from app.services.document_exceptions import DocumentParsingError
from app.services.parsers.base import BaseDocumentParser


class TxtParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> str:
        try:
            # utf-8-sig strips a BOM if present; latin-1 is a last-resort
            # fallback so we never crash on legacy Windows-exported text.
            try:
                return file_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                return file_path.read_text(encoding="latin-1")
        except OSError as exc:
            raise DocumentParsingError(
                f"Failed to read text file '{file_path.name}': {exc}"
            ) from exc
