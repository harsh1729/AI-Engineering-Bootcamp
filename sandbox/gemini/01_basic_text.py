"""Basic non-streaming text generation with generateContent."""

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
    response = client.models.generate_content(
        model=MODEL,
        contents="Explain what an LLM is in two sentences.",
    )

    print(response.text)


if __name__ == "__main__":
    main()
