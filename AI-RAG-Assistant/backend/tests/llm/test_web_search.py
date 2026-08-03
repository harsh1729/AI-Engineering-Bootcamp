from unittest.mock import MagicMock, patch

import pytest

from llm_sdk.services.web_search_service import WebSearchService
from llm_sdk.tool_functions.web_search import (
    WEB_SEARCH_UNAVAILABLE_MESSAGE,
    web_search,
)


class TestWebSearchService:
    def test_formats_search_results(self) -> None:
        service = WebSearchService()

        with patch("llm_sdk.services.web_search_service.DDGS") as mock_ddgs_cls:
            mock_ddgs = MagicMock()
            mock_ddgs.__enter__.return_value = mock_ddgs
            mock_ddgs.text.return_value = [
                {
                    "title": "India medal tally",
                    "href": "https://example.com/medals",
                    "body": "India has 5 gold medals.",
                },
                {
                    "title": "Live scores",
                    "href": "https://example.com/scores",
                    "body": "IND 240/3",
                },
            ]
            mock_ddgs_cls.return_value = mock_ddgs

            result = service.search("India medal tally Olympics")

        assert "[1] India medal tally" in result
        assert "https://example.com/medals" in result
        assert "India has 5 gold medals." in result
        assert "[2] Live scores" in result
        mock_ddgs.text.assert_called_once_with(
            "India medal tally Olympics",
            max_results=5,
        )

    def test_returns_message_when_no_results(self) -> None:
        service = WebSearchService()

        with patch("llm_sdk.services.web_search_service.DDGS") as mock_ddgs_cls:
            mock_ddgs = MagicMock()
            mock_ddgs.__enter__.return_value = mock_ddgs
            mock_ddgs.text.return_value = []
            mock_ddgs_cls.return_value = mock_ddgs

            result = service.search("obscure query with no hits")

        assert result == "No web results found for this query."

    def test_rejects_empty_query(self) -> None:
        service = WebSearchService()

        with pytest.raises(ValueError, match="must not be empty"):
            service.search("   ")


class TestWebSearchTool:
    def test_returns_unavailable_message_when_service_fails(self) -> None:
        with patch(
            "llm_sdk.tool_functions.web_search.web_search_service.search",
            side_effect=RuntimeError("network down"),
        ):
            result = web_search("current RBI governor")

        assert result == WEB_SEARCH_UNAVAILABLE_MESSAGE
