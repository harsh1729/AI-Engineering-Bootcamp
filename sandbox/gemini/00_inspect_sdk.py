"""Inspect the google-genai SDK and a single generateContent response."""

import importlib.metadata
import os
import sys

from google import genai

MODEL = "gemini-3.5-flash"


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"google-genai version: {importlib.metadata.version('google-genai')}")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents="Say hello in one word.",
    )

    print(f"response type: {type(response).__name__}")
    print(f"response.text: {response.text!r}")

    if response.candidates:
        candidate = response.candidates[0]
        print(f"candidates[0].finish_reason: {candidate.finish_reason}")

        if candidate.content and candidate.content.parts:
            print(f"candidates[0].content.parts count: {len(candidate.content.parts)}")
            for index, part in enumerate(candidate.content.parts):
                print(f"  parts[{index}] type: {type(part).__name__}")
                if part.text:
                    print(f"  parts[{index}].text: {part.text!r}")

    if response.usage_metadata:
        usage = response.usage_metadata
        print(f"usage_metadata.prompt_token_count: {usage.prompt_token_count}")
        print(f"usage_metadata.candidates_token_count: {usage.candidates_token_count}")
        print(f"usage_metadata.total_token_count: {usage.total_token_count}")


if __name__ == "__main__":
    main()
