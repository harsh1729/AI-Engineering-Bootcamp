import re
from typing import Callable

# `(header_line, body)` — header_line may be empty for preamble content.
HeaderSection = tuple[str, str]
HeaderDetector = Callable[[str], list[HeaderSection]]

_MARKDOWN_HEADER_LINE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
_HTML_HEADER_LINE = re.compile(
    r"^\s*(<h[1-6][^>]*>.*?</h[1-6]>)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def segment_by_headers(text: str) -> list[HeaderSection]:
    """Split text into header sections using the first matching detector.

    Returns an empty list when no known heading pattern is found, allowing
    callers to fall back to generic chunking. Additional detectors (for example
    DOCX- or PDF-derived heading markers injected during parsing) can be
    registered in `_HEADER_DETECTORS`.
    """
    for detector in _HEADER_DETECTORS:
        sections = detector(text)
        if sections:
            return sections
    return []


def _segment_by_line_pattern(
    text: str,
    header_pattern: re.Pattern[str],
) -> list[HeaderSection]:
    """Shared helper for single-line heading patterns."""
    matches = list(header_pattern.finditer(text))
    if not matches:
        return []

    sections: list[HeaderSection] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for index, match in enumerate(matches):
        header = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((header, body))

    return sections


def _segment_by_markdown_headers(text: str) -> list[HeaderSection]:
    """Detect ATX markdown headings (`#` through `######`)."""
    return _segment_by_line_pattern(text, _MARKDOWN_HEADER_LINE)


def _segment_by_html_headers(text: str) -> list[HeaderSection]:
    """Detect HTML heading tags (`<h1>` through `<h6>`)."""
    return _segment_by_line_pattern(text, _HTML_HEADER_LINE)


_HEADER_DETECTORS: tuple[HeaderDetector, ...] = (
    _segment_by_markdown_headers,
    _segment_by_html_headers,
)
