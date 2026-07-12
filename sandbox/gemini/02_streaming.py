"""Streaming text generation with generate_content_stream."""

import os
import sys

from google import genai

MODEL = "gemini-3.5-flash"


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Streaming response:")
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents="Write a haiku about backend engineering.",
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
