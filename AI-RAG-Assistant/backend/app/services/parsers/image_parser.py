from pathlib import Path

import pytesseract
from PIL import Image

from app.services.document_exceptions import DocumentParsingError
from app.services.parsers.base import BaseDocumentParser


class ImageParser(BaseDocumentParser):
    """Extract indexable text from raster images using local OCR."""

    def parse(self, file_path: Path) -> str:
        try:
            with Image.open(file_path) as image:
                prepared = image.convert("RGB")
                extracted_text = pytesseract.image_to_string(prepared)
        except DocumentParsingError:
            raise
        except OSError as exc:
            raise DocumentParsingError(
                f"Failed to read image file '{file_path.name}': {exc}"
            ) from exc
        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to extract text from image '{file_path.name}': {exc}"
            ) from exc

        normalized = extracted_text.strip()
        if not normalized:
            raise DocumentParsingError(
                f"No text could be extracted from image '{file_path.name}' via OCR."
            )

        return normalized
