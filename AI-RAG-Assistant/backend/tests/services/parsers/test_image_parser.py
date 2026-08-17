from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from app.services.document_exceptions import DocumentParsingError
from app.services.parsers.image_parser import ImageParser


@pytest.fixture
def parser() -> ImageParser:
    return ImageParser()


class TestImageParser:
    def test_extracts_trimmed_ocr_text(self, parser: ImageParser, tmp_path: Path) -> None:
        image_path = tmp_path / "scan.png"
        image_path.write_bytes(b"fake-png")

        with patch("app.services.parsers.image_parser.Image.open") as mock_open, patch(
            "app.services.parsers.image_parser.pytesseract.image_to_string",
            return_value="  Secret code BLUE42  \n",
        ):
            mock_image = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_image
            mock_image.convert.return_value = mock_image

            text = parser.parse(image_path)

        assert text == "Secret code BLUE42"
        mock_image.convert.assert_called_once_with("RGB")

    def test_raises_when_ocr_returns_empty_text(
        self,
        parser: ImageParser,
        tmp_path: Path,
    ) -> None:
        image_path = tmp_path / "blank.png"
        image_path.write_bytes(b"fake-png")

        with patch("app.services.parsers.image_parser.Image.open") as mock_open, patch(
            "app.services.parsers.image_parser.pytesseract.image_to_string",
            return_value="   \n",
        ):
            mock_image = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_image
            mock_image.convert.return_value = mock_image

            with pytest.raises(DocumentParsingError, match="No text could be extracted"):
                parser.parse(image_path)

    def test_wraps_unexpected_failures(self, parser: ImageParser, tmp_path: Path) -> None:
        image_path = tmp_path / "broken.png"
        image_path.write_bytes(b"fake-png")

        with patch(
            "app.services.parsers.image_parser.Image.open",
            side_effect=OSError("corrupt image"),
        ):
            with pytest.raises(DocumentParsingError, match="Failed to read image file"):
                parser.parse(image_path)
