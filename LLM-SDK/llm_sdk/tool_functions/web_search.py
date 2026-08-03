import logging

from llm_sdk.models.tools import LLMTool, LLMToolParam
from llm_sdk.services.web_search_service import WebSearchService

logger = logging.getLogger(__name__)

web_search_service = WebSearchService()

WEB_SEARCH_UNAVAILABLE_MESSAGE = (
    "Web search is unavailable. Respond plainly that you do not have "
    "up-to-date information for this question."
)


def web_search(query: str) -> str:
    try:
        return web_search_service.search(query)
    except Exception:
        logger.exception("web_search failed for query=%r", query)
        return WEB_SEARCH_UNAVAILABLE_MESSAGE


WEB_SEARCH_TOOL = LLMTool(
    name="web_search",
    description=(
        "Search the web for up-to-date information such as live sports scores, "
        "current office holders, medal tallies, breaking news, or any facts that "
        "may have changed recently."
    ),
    parameters=[
        LLMToolParam(
            name="query",
            type="string",
            description=(
                "Focused search query, e.g. 'India medal tally Olympics 2026' "
                "or 'current finance minister of India'."
            ),
        ),
    ],
)
