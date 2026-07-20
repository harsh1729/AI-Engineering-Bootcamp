from abc import ABC, abstractmethod
from pathlib import Path


class BaseDocumentParser(ABC):
    """Strategy interface for extracting plain text from a single file format."""

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Return the extracted text for `file_path`.

        Implementations should raise DocumentParsingError for corrupt or
        unreadable files, and UnsupportedDocumentTypeError when the format
        is intentionally not handled (e.g. legacy binary Office formats).
        """
