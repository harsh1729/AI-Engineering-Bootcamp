import re

_TABLE_HEADER_PATTERN = re.compile(r"TABLE \d+:")
# A new table block begins at document start or after a blank line when TABLE appears on that line.
_TABLE_BLOCK_START_PATTERN = re.compile(r"(?=(?:^|\n\n)[^\n]*TABLE \d+:)")


def segment_by_table_blocks(text: str) -> list[str]:
    """Split text into segments, starting new segments at table block boundaries."""
    if not _TABLE_HEADER_PATTERN.search(text):
        return [text]

    parts = _TABLE_BLOCK_START_PATTERN.split(text)
    return [part for part in parts if part]


def is_table_segment(segment: str) -> bool:
    """Return True when the segment begins with a TABLE N: header line."""
    if not segment.strip():
        return False

    first_line = segment.lstrip().split("\n", 1)[0]
    return _TABLE_HEADER_PATTERN.search(first_line) is not None
