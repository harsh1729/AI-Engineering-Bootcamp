"""Manual two-turn function calling with generateContent."""

import os
import sys
from datetime import datetime, timezone as tz

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"

GET_CURRENT_TIME_DECLARATION = {
    "name": "get_current_time",
    "description": "Returns the current UTC time labeled with the requested timezone name.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone label, e.g. UTC or America/New_York",
            }
        },
        "required": ["timezone"],
    },
}


def get_current_time(timezone: str) -> dict:
    return {
        "timezone": timezone,
        "utc_time": datetime.now(tz.utc).isoformat(),
    }


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    tool = types.Tool(function_declarations=[GET_CURRENT_TIME_DECLARATION])
    config = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents = ["What is the current time in UTC?"]

    # Turn 1: model requests a function call.
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    function_call = None
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call = part.function_call
            break

    if function_call is None:
        print("Model did not request a function call.")
        print(response.text)
        return

    print(f"Function requested: {function_call.name}")
    print(f"Arguments: {dict(function_call.args)}")
    print(f"Call ID: {function_call.id}")

    result = get_current_time(**dict(function_call.args))
    print(f"Local result: {result}")

    # Turn 2: append model turn and function result, then continue.
    contents.append(response.candidates[0].content)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=function_call.name,
                        response=result,
                        id=function_call.id,
                    )
                )
            ],
        )
    )

    final_response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    print(f"\nFinal response:\n{final_response.text}")


if __name__ == "__main__":
    main()
