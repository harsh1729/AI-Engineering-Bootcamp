from pathlib import Path

from app.services.document_exceptions import DocumentParsingError
from app.services.parsers.base import BaseDocumentParser


class PdfParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> str:
        try:
            # Imported inside parse() so a missing PyMuPDF install cannot take
            # down the entire FastAPI app (upload / chat / other parsers).
            import fitz
        except ImportError as exc:
            raise DocumentParsingError(
                "PDF parsing requires PyMuPDF. Install it in the backend venv "
                "with: pip install PyMuPDF"
            ) from exc

        try:
            # "text" + sort=True lays out blocks in visual reading order, which
            # reduces the broken spacing common with pypdf on multi-column /
            # resume-style PDFs. No OCR - text layer only.
            document = fitz.open(file_path)
            try:
                pages: list[str] = []
                for page in document:
                    text = page.get_text("text", sort=True) or ""
                    if text.strip():
                        pages.append(text.strip())
                return "\n\n".join(pages).strip()
            finally:
                document.close()
        except DocumentParsingError:
            raise
        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to parse PDF '{file_path.name}': {exc}"
            ) from exc
