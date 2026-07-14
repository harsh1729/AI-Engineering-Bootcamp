"""Explore streaming + manual tool calling with generate_content_stream.

Phase 0 goal: answer Milestone 4 design questions by inspecting raw stream chunks.

Scenarios:
- Single tool: UTC time query (simple baseline)
- Multi tool: London weather + time + currency (matches production main.py)

Questions this script investigates:
1. When do function_call parts appear (early vs final chunks)?
2. Are stream chunks delta or cumulative?
3. What finish_reason is set on a tool-request turn?
4. Are function_call args complete in a single chunk?
5. Does accessing chunk.text warn on tool-only turns?
6. Can multiple function_calls arrive in one stream turn?
"""

import os
import sys
from datetime import datetime, timezone as tz
from typing import Callable

from google import genai
from google.genai import types

# Match production primary model for realistic Milestone 4 behavior.
MODEL = "gemini-3.1-flash-lite"

# --- Tool declarations (sandbox-local; no imports from 01-LLM-SDK) ---

GET_CURRENT_TIME_DECLARATION = {
    "name": "get_current_time",
    "description": "Returns the current UTC time labeled with the requested timezone name.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone such as Europe/London or UTC.",
            }
        },
        "required": ["timezone"],
    },
}

GET_WEATHER_DECLARATION = {
    "name": "get_weather",
    "description": "Returns the current weather for the specified location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location such as London or New York.",
            }
        },
        "required": ["location"],
    },
}

CONVERT_CURRENCY_DECLARATION = {
    "name": "convert_currency",
    "description": "Converts an amount from one currency to another.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "Amount to convert.",
            },
            "from_currency": {
                "type": "string",
                "description": "Source currency code such as USD.",
            },
            "to_currency": {
                "type": "string",
                "description": "Target currency code such as GBP.",
            },
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}

SINGLE_TOOL_DECLARATIONS = [GET_CURRENT_TIME_DECLARATION]

MULTI_TOOL_DECLARATIONS = [
    GET_CURRENT_TIME_DECLARATION,
    GET_WEATHER_DECLARATION,
    CONVERT_CURRENCY_DECLARATION,
]


# --- Stub tool implementations ---

def get_current_time(timezone: str) -> dict:
    return {
        "timezone": timezone,
        "utc_time": datetime.now(tz.utc).isoformat(),
    }


def get_weather(location: str) -> str:
    return f"Weather in {location}: 23.5°C, partly cloudy (sandbox stub)."


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    rates = {("USD", "GBP"): 0.7469, ("USD", "INR"): 83.0}
    rate = rates.get((from_currency.upper(), to_currency.upper()), 1.0)
    converted = round(amount * rate, 2)
    return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (sandbox stub)."


TOOL_HANDLERS: dict[str, Callable] = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "convert_currency": convert_currency,
}


def extract_text_from_parts(parts: list[types.Part]) -> str:
    return "".join(part.text for part in parts if part.text)


def describe_parts(parts: list[types.Part]) -> list[str]:
    descriptions = []

    for index, part in enumerate(parts):
        if part.text:
            descriptions.append(f"parts[{index}].text={part.text!r}")
        if part.function_call:
            fc = part.function_call
            descriptions.append(
                f"parts[{index}].function_call name={fc.name!r} "
                f"id={fc.id!r} args={dict(fc.args)}"
            )
        if part.thought:
            descriptions.append(f"parts[{index}].thought={part.thought!r}")
        if not part.text and not part.function_call and not part.thought:
            descriptions.append(f"parts[{index}] (no text/function_call/thought)")

    return descriptions


def inspect_stream(
    stream,
    *,
    label: str,
    probe_chunk_text: bool = True,
) -> tuple[list[types.GenerateContentResponse], list[types.FunctionCall], types.Content | None]:

    print(f"\n{'=' * 72}")
    print(label)
    print(f"{'=' * 72}")

    chunks: list[types.GenerateContentResponse] = []
    function_calls: list[types.FunctionCall] = []
    seen_call_ids: set[str] = set()
    assistant_parts: list[types.Part] = []
    last_content: types.Content | None = None
    previous_text_snapshot = ""

    for index, chunk in enumerate(stream):
        chunks.append(chunk)

        candidate = chunk.candidates[0] if chunk.candidates else None
        finish_reason = candidate.finish_reason if candidate else None
        parts = candidate.content.parts if candidate and candidate.content else []
        part_text = extract_text_from_parts(parts)

        print(f"\n--- chunk {index} ---")
        print(f"response_id: {chunk.response_id!r}")
        print(f"finish_reason: {finish_reason}")

        if probe_chunk_text and index == 0:
            print("Probing chunk.text on first chunk (watch for SDK warning above)...")
            print(f"chunk.text: {chunk.text!r}")

        print(f"parts count: {len(parts)}")
        for line in describe_parts(parts):
            print(f"  {line}")

        if part_text != previous_text_snapshot:
            if part_text.startswith(previous_text_snapshot):
                delta = part_text[len(previous_text_snapshot):]
                print(f"text mode: cumulative (+{len(delta)} chars delta)")
                if delta:
                    print(f"text delta: {delta!r}")
            else:
                print("text mode: delta/incremental (this chunk's text only)")
                print(f"chunk text: {part_text!r}")
            previous_text_snapshot = part_text

        for part in parts:
            if part.function_call:
                call_id = part.function_call.id
                if call_id and call_id not in seen_call_ids:
                    seen_call_ids.add(call_id)
                    function_calls.append(part.function_call)
                assistant_parts.append(part)

        if candidate and candidate.content:
            last_content = candidate.content

    assistant_content = (
        types.Content(role="model", parts=assistant_parts)
        if assistant_parts
        else None
    )
    content_for_follow_up = assistant_content or last_content

    print(f"\n--- {label} summary ---")
    print(f"total chunks: {len(chunks)}")
    print(f"unique function_calls collected: {len(function_calls)}")
    for fc in function_calls:
        print(f"  - {fc.name} id={fc.id!r} args={dict(fc.args)}")

    print(
        f"assistant_content source: "
        f"{'merged function_call parts' if assistant_content else 'last chunk'}"
    )

    if content_for_follow_up is not None:
        print("content to append for follow-up:")
        print(f"  role: {content_for_follow_up.role!r}")
        print(f"  parts count: {len(content_for_follow_up.parts)}")
        for line in describe_parts(content_for_follow_up.parts):
            print(f"  {line}")

    return chunks, function_calls, content_for_follow_up


def execute_tool(function_call: types.FunctionCall) -> dict:
    handler = TOOL_HANDLERS.get(function_call.name)
    if handler is None:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown tool: {function_call.name}",
        }

    try:
        result = handler(**dict(function_call.args))
        return {"success": True, "result": result, "error": None}
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def build_function_response_content(
    function_calls: list[types.FunctionCall],
) -> types.Content:

    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name=function_call.name,
                    response=execute_tool(function_call),
                    id=function_call.id,
                )
            )
            for function_call in function_calls
        ],
    )


def run_streaming_tool_scenario(
    client: genai.Client,
    *,
    label: str,
    prompt: str,
    declarations: list[dict],
) -> None:

    print(f"\n{'#' * 72}")
    print(f"SCENARIO: {label}")
    print(f"{'#' * 72}")
    print(f"Prompt: {prompt!r}")

    tool = types.Tool(function_declarations=declarations)
    config = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list = [prompt]

    turn_1_stream = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=config,
    )
    _, function_calls, assistant_content = inspect_stream(
        turn_1_stream,
        label=f"{label} — TURN 1 tool request stream",
    )

    if not function_calls:
        print("\nModel did not request a function call during streaming.")
        return

    if assistant_content is None:
        print("Error: could not capture assistant_content from stream.", file=sys.stderr)
        sys.exit(1)

    print("\nExecuting tools:")
    for function_call in function_calls:
        result = execute_tool(function_call)
        print(f"  - {function_call.name}({dict(function_call.args)}) -> {result}")

    contents.append(assistant_content)
    contents.append(build_function_response_content(function_calls))

    print("\nStreaming final answer:")
    turn_2_stream = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=config,
    )
    inspect_stream(
        turn_2_stream,
        label=f"{label} — TURN 2 final answer stream",
        probe_chunk_text=False,
    )


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    run_streaming_tool_scenario(
        client,
        label="single tool",
        prompt="What is the current time in UTC?",
        declarations=SINGLE_TOOL_DECLARATIONS,
    )

    run_streaming_tool_scenario(
        client,
        label="multi tool",
        prompt=(
            "How is the weather in London? Also tell current time and convert 100 USD to its currency."
        ),
        declarations=MULTI_TOOL_DECLARATIONS,
    )

    print("\nDone. Review chunk logs above for Milestone 4 design decisions.")
    print_milestone_4_findings()


def print_milestone_4_findings() -> None:
    print(f"\n{'=' * 72}")
    print("MILESTONE 4 FINDINGS (from this sandbox run)")
    print(f"{'=' * 72}")
    print("1. function_call appears in early chunk(s), complete with id + args.")
    print("2. A final empty STOP chunk follows — do NOT use last chunk as assistant_content.")
    print("3. Merge all function_call parts across chunks into one model Content.")
    print("4. Multiple tools: send multiple FunctionResponse parts in one user Content.")
    print("5. Text answer chunks are delta/incremental — yield part.text directly per chunk.")
    print("6. Accessing chunk.text on tool-only chunks triggers SDK warning — use parts instead.")
    print("7. finish_reason=STOP arrives on a separate trailing chunk, not with function_call.")


if __name__ == "__main__":
    main()
