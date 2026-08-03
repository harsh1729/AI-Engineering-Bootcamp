from llm_sdk.enums import MessageRole
from llm_sdk.models import LLMMessage, LLMRequest

from app.llm.chat_tools import DEFAULT_CHAT_TOOLS
from app.models.chat import ChatRequest

TOOLS_SYSTEM_PROMPT = (
    "You have tools for current time, weather, currency conversion, and web search. "
    "Use them when the user asks for live, location-specific, or up-to-date information. "
    "Use web_search for current events, live sports scores, medal tallies, office holders, "
    "or anything that may have changed after your training cutoff. "
    "Never show raw JSON, tool names, or tool payloads in your reply. "
    "If web_search is unavailable or returns no useful results, respond plainly that you "
    "do not have up-to-date information for that question."
)


def build_chat_llm_request(
    request: ChatRequest,
    *,
    include_tools: bool = False,
) -> LLMRequest:
    """Build an LLMRequest from a chat request, optionally attaching SDK tools."""
    messages = [
        LLMMessage(role=message.role, content=message.content)
        for message in request.messages
    ]
    tools: list = []

    if include_tools:
        tools = list(DEFAULT_CHAT_TOOLS)
        has_system_message = any(
            message.role == MessageRole.SYSTEM for message in messages
        )
        if not has_system_message:
            messages = [
                LLMMessage(role=MessageRole.SYSTEM, content=TOOLS_SYSTEM_PROMPT),
                *messages,
            ]

    return LLMRequest(
        messages=messages,
        provider=request.provider,
        model=request.model,
        tools=tools,
    )
