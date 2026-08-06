import pytest

from app.chunking.chunk_size_utils import (
    merge_text_pieces,
    split_by_character_window,
    split_into_sentences,
    validate_chunk_params,
)
from app.chunking.header_segmentation import segment_by_headers


class TestValidateChunkParams:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(0, 0), (-1, 0)],
    )
    def test_rejects_non_positive_chunk_size(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            validate_chunk_params(chunk_size, chunk_overlap)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            validate_chunk_params(100, -1)

    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            validate_chunk_params(10, 10)


class TestSplitIntoSentences:
    def test_empty_text_returns_empty_list(self) -> None:
        assert split_into_sentences("") == []

    def test_single_sentence_without_trailing_space(self) -> None:
        assert split_into_sentences("Hello world") == ["Hello world"]

    def test_splits_on_period_question_and_exclamation(self) -> None:
        text = "First sentence. Second one? Third!"
        assert split_into_sentences(text) == [
            "First sentence. ",
            "Second one? ",
            "Third!",
        ]

    def test_does_not_split_on_common_abbreviations(self) -> None:
        text = "Dr. Smith lives in New York. He works at OpenAI Inc."
        assert split_into_sentences(text) == [
            "Dr. Smith lives in New York. ",
            "He works at OpenAI Inc.",
        ]

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Mr. and Mrs. Smith arrived. They checked in.", [
                "Mr. and Mrs. Smith arrived. ",
                "They checked in.",
            ]),
            ("See e.g. the prior section. Next topic here.", [
                "See e.g. the prior section. ",
                "Next topic here.",
            ]),
        ],
    )
    def test_skips_listed_abbreviations(self, text: str, expected: list[str]) -> None:
        assert split_into_sentences(text) == expected


class TestSegmentByHeaders:
    def test_returns_empty_list_when_no_headers(self) -> None:
        assert segment_by_headers("Plain paragraph.\nAnother line.") == []

    def test_splits_multiple_markdown_header_levels(self) -> None:
        text = (
            "Preamble text.\n\n"
            "# Title\n"
            "Intro body.\n\n"
            "## Section A\n"
            "Section A body.\n\n"
            "### Subsection\n"
            "Nested body."
        )
        sections = segment_by_headers(text)

        assert sections == [
            ("", "Preamble text."),
            ("# Title", "Intro body."),
            ("## Section A", "Section A body."),
            ("### Subsection", "Nested body."),
        ]

    def test_splits_html_headers(self) -> None:
        text = "<h1>Title</h1>\n\nIntro.\n\n<h2>Details</h2>\n\nMore info."
        sections = segment_by_headers(text)

        assert sections == [
            ("<h1>Title</h1>", "Intro."),
            ("<h2>Details</h2>", "More info."),
        ]
