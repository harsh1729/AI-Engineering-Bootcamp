from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_sdk.models import LLMRequest, LLMResponseChunk
from llm_sdk.models.tools import LLMToolCall, LLMToolExecutionResult
from llm_sdk.providers import openai_provider as openai_provider_module
from llm_sdk.providers.openai_provider import OpenAIProvider
from llm_sdk.serializers.openai_request_serializer import OpenAIRequestSerializer
from llm_sdk.serializers.tool_output import format_tool_output_for_llm
from llm_sdk.tool_functions import WEB_SEARCH_TOOL


class TestFormatToolOutputForLlm:
    def test_returns_plain_result_text(self) -> None:
        tool_result = LLMToolExecutionResult(
            tool_call=LLMToolCall(
                call_id="call-1",
                name="web_search",
                arguments={"query": "medal tally"},
            ),
            result="[1] Live scores\nURL: https://example.com",
        )

        assert format_tool_output_for_llm(tool_result) == (
            "[1] Live scores\nURL: https://example.com"
        )

    def test_returns_error_message(self) -> None:
        tool_result = LLMToolExecutionResult(
            tool_call=LLMToolCall(
                call_id="call-1",
                name="get_weather",
                arguments={"location": "London"},
            ),
            error="service unavailable",
        )

        assert format_tool_output_for_llm(tool_result) == (
            "Tool error: service unavailable"
        )


class TestOpenAIToolOutputSerialization:
    def test_uses_plain_text_output(self) -> None:
        serializer = OpenAIRequestSerializer()
        request = LLMRequest(messages=[], tools=[WEB_SEARCH_TOOL])
        tool_result = LLMToolExecutionResult(
            tool_call=LLMToolCall(
                call_id="call-1",
                name="web_search",
                arguments={"query": "medal tally"},
            ),
            result="No web results found for this query.",
        )

        payload = serializer.serialize_tool_results(
            request=request,
            previous_response_id="resp-1",
            tool_results=[tool_result],
        )

        assert payload["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "No web results found for this query.",
            }
        ]


class TestOpenAIStreamToolTurns:
    def test_discards_text_from_tool_call_turn_and_streams_final_answer(self) -> None:
        class FakeCreatedEvent:
            pass

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.request_serializer = OpenAIRequestSerializer()
        provider.response_chunk_serializer = MagicMock()

        created_event = FakeCreatedEvent()
        created_event.response = SimpleNamespace(id="resp-1")
        first_stream = [created_event, object(), object()]
        second_stream = [object()]

        provider.response_chunk_serializer.serialize.side_effect = [
            LLMResponseChunk(text="I'll broaden the search."),
            LLMResponseChunk(
                tool_call=LLMToolCall(
                    call_id="call-1",
                    name="web_search",
                    arguments={"query": "Commonwealth Games medal table live"},
                )
            ),
            LLMResponseChunk(text="India currently has 12 medals."),
        ]

        provider._execute_tools = MagicMock(
            return_value=[
                LLMToolExecutionResult(
                    tool_call=LLMToolCall(
                        call_id="call-1",
                        name="web_search",
                        arguments={"query": "Commonwealth Games medal table live"},
                    ),
                    result="[1] Medal table",
                )
            ]
        )
        provider._call_with_retry = MagicMock(return_value=second_stream)

        request = LLMRequest(messages=[], tools=[WEB_SEARCH_TOOL])

        with patch.object(
            openai_provider_module,
            "ResponseCreatedEvent",
            FakeCreatedEvent,
        ):
            chunks = list(provider._finalize_stream(request, iter(first_stream)))

        assert chunks == [LLMResponseChunk(text="India currently has 12 medals.")]
        provider._execute_tools.assert_called_once()
