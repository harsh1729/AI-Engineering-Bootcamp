from llm_sdk.models.tools import LLMToolExecutionResult


def format_tool_output_for_llm(tool_result: LLMToolExecutionResult) -> str:
    """Return plain text tool output for provider follow-up requests."""
    if tool_result.error is not None:
        return f"Tool error: {tool_result.error}"
    if tool_result.result is None:
        return ""
    return str(tool_result.result)
