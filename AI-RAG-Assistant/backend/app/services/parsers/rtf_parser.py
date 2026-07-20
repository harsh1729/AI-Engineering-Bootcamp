from pathlib import Path

from striprtf.striprtf import rtf_to_text

from app.services.document_exceptions import DocumentParsingError
from app.services.parsers.base import BaseDocumentParser


class RtfParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> str:
        try:
            try:
                raw = file_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                raw = file_path.read_text(encoding="latin-1")
        except OSError as exc:
            raise DocumentParsingError(
                f"Failed to read RTF '{file_path.name}': {exc}"
            ) from exc

        # striprtf is permissive on random text; require a real RTF header so
        # clearly non-RTF / malformed payloads fail loudly instead of returning
        # garbage as "parsed" content.
        if not raw.lstrip().lower().startswith(r"{\rtf"):
            raise DocumentParsingError(
                f"Failed to parse RTF '{file_path.name}': file is not valid RTF."
            )

        try:
            return rtf_to_text(raw).strip()
        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to parse RTF '{file_path.name}': {exc}"
            ) from exc
