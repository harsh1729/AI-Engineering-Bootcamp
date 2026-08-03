from unittest.mock import MagicMock

import pytest

from llm_sdk.models.tools import LLMToolCall
from llm_sdk.providers.llm_provider import LLMProvider
from llm_sdk.tool_functions import ToolRegistry
from llm_sdk.tool_functions.web_search import WEB_SEARCH_UNAVAILABLE_MESSAGE


class _DummyProvider(LLMProvider):
    def generate_response(self, request):
        raise NotImplementedError

    def generate_stream(self, request):
        raise NotImplementedError


class TestExecuteTools:
    def test_web_search_timeout_returns_unavailable_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _DummyProvider()

        def raise_timeout(tool, arguments):
            raise TimeoutError("timed out")

        monkeypatch.setattr(provider, "_execute_tool_with_timeout", raise_timeout)

        results = provider._execute_tools(
            [
                LLMToolCall(
                    call_id="call-1",
                    name="web_search",
                    arguments={"query": "live cricket score"},
                )
            ]
        )

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].result == WEB_SEARCH_UNAVAILABLE_MESSAGE

    def test_other_tools_still_return_error_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _DummyProvider()

        def broken_tool(**kwargs: object) -> str:
            raise RuntimeError("service unavailable")

        monkeypatch.setitem(ToolRegistry._tools, "get_weather", broken_tool)

        results = provider._execute_tools(
            [
                LLMToolCall(
                    call_id="call-2",
                    name="get_weather",
                    arguments={"location": "London"},
                )
            ]
        )

        assert results[0].result is None
        assert results[0].error == "service unavailable"

    def test_tool_timeout_is_twenty_seconds(self) -> None:
        from llm_sdk.providers.llm_provider import TOOL_TIMEOUT_SECONDS

        assert TOOL_TIMEOUT_SECONDS == 20
