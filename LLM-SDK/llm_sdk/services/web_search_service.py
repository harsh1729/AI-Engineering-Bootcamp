from ddgs import DDGS

from llm_sdk.config import WEB_SEARCH_MAX_RESULTS


class WebSearchService:
    """Runs text search via DuckDuckGo and formats snippets for the LLM."""

    def search(self, query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query must not be empty.")

        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    normalized_query,
                    max_results=WEB_SEARCH_MAX_RESULTS,
                )
            )

        if not raw_results:
            return "No web results found for this query."

        return self._format_results(raw_results)

    def _format_results(self, results: list[dict]) -> str:
        formatted_blocks: list[str] = []

        for index, result in enumerate(results, start=1):
            title = result.get("title") or "Untitled"
            url = result.get("href") or result.get("link") or "unknown"
            snippet = result.get("body") or result.get("snippet") or ""
            formatted_blocks.append(
                f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}"
            )

        return "\n\n".join(formatted_blocks)
