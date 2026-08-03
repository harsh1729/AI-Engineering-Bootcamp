from dotenv import load_dotenv
import os
from llm_sdk.enums import ProviderType, GeminiThinkingLevel

load_dotenv()

# Configuration
# Note: keys are intentionally not validated here. Each provider validates
# only the key it actually needs, when it is instantiated, so importing the
# SDK or using one provider doesn't require every provider's key to be set.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"
GEMINI_THINKING_LEVEL = GeminiThinkingLevel.MINIMAL

LLM_MAX_RETRIES = 2
LLM_RETRY_BASE_DELAY_SECONDS = 1.0

WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

LLM_PROVIDER = ProviderType.GEMINI  

    