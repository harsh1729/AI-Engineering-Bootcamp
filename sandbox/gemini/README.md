# Gemini Learning Sandbox

This folder is an isolated learning environment for the official Google Gen AI SDK (`google-genai`) and the **generateContent** API. It exists separately from the production LLM SDK in `01-LLM-SDK/`.

## Architectural purpose

Before building a `GeminiProvider`, serializers, and factory wiring, this sandbox answers a different question: **how does Google's API actually work?**

The production SDK wraps vendor APIs behind unified models (`LLMRequest`, `LLMResponse`) and provider-owned orchestration. That abstraction is useful, but it hides vendor-specific details that directly affect serializer design:

- How requests are shaped (`contents`, `parts`, `GenerateContentConfig`)
- How responses are structured (`candidates`, `usage_metadata`, `finish_reason`)
- How tool calling works across multiple turns (`function_call` → local execution → `function_response`)
- How streaming delivers partial output

This sandbox deliberately stays **outside** the production architecture so you can:

1. **Learn the vendor API on its own terms** — no `LLMProvider`, no serializers, no factory.
2. **De-risk provider design** — confirm generateContent behavior before writing production code.
3. **Keep production code stable** — OpenAI and Claude providers remain untouched during exploration.
4. **Inform serializer decisions** — observations from `00_inspect_sdk.py` guide what `GeminiRequestSerializer` and `GeminiResponseSerializer` must map later.

```
sandbox/gemini/          ← vendor exploration (this folder)
01-LLM-SDK/providers/    ← production orchestration (later phases)
01-LLM-SDK/serializers/  ← request/response mapping (later phases)
```

## Prerequisites

1. Install the SDK into the project virtual environment:
   ```bash
   pip install google-genai
   ```
2. Export your API key (no `.env` loading in these scripts):
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Recommended run order

Run the scripts in numbered order from the repo root:

```bash
python sandbox/gemini/00_inspect_sdk.py
python sandbox/gemini/01_basic_text.py
python sandbox/gemini/02_streaming.py
python sandbox/gemini/03_tool_calling.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `00_inspect_sdk.py` | **Start here.** Prints the installed SDK version and inspects a single `generate_content` response: type, `text`, `candidates`, `parts`, `usage_metadata`, and `finish_reason`. |
| `01_basic_text.py` | Minimal non-streaming text generation. |
| `02_streaming.py` | Streams output with `generate_content_stream`, printing text as it arrives. |
| `03_tool_calling.py` | Manual two-turn function calling: disables SDK automatic function calling, executes a local function, sends the result back, and prints the final model response. |

## Conventions

- API surface: **generateContent only** (`client.models.generate_content`) — not the Interactions API.
- Authentication: `os.getenv("GEMINI_API_KEY")`.
- Model: `gemini-3.5-flash`.
- No imports from `01-LLM-SDK` production modules.

## After validation

Once all scripts run successfully, pin the installed `google-genai` version in `01-LLM-SDK/requirements.txt`. Production SDK integration (provider, serializers, factory) happens in later phases.
