from dotenv import load_dotenv
import os
from enums import ProviderType

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not configured")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not configured")

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-opus-4-8"

LLM_PROVIDER = ProviderType.OPENAI  

    