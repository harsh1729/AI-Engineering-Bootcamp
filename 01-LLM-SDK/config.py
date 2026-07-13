from dotenv import load_dotenv
import os
from enums import ProviderType, GeminiThinkingLevel

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not configured")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not configured")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured")

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"
GEMINI_THINKING_LEVEL = GeminiThinkingLevel.MINIMAL

LLM_MAX_RETRIES = 2
LLM_RETRY_BASE_DELAY_SECONDS = 1.0

LLM_PROVIDER = ProviderType.GEMINI  

    