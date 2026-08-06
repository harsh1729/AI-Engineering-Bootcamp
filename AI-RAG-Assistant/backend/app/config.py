from pathlib import Path
import os

from dotenv import load_dotenv

# Resolved relative to this file (not the process CWD) so uploads always land
# in the same place regardless of where uvicorn is launched from.
BACKEND_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(BACKEND_ROOT / ".env")

# Maximum number of user messages a single guest may send during the demo.
# Enforced by app.services.usage_tracker.
MAX_DEMO_INTERACTIONS = 20

# Maximum size of a single uploaded document. Enforced by the upload endpoint
# before the file is written to disk (and mirrored client-side for UX).
MAX_DOCUMENT_SIZE_MB = 15
MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024

UPLOAD_DIR = BACKEND_ROOT / "uploads"

# uploads/ is a root with purpose-specific subfolders:
# - files/  finalized document uploads (used today by document_storage).
# - images/ reserved for a future image-upload feature (not wired up yet).
# - temp/   reserved for future staged/in-progress uploads (not wired up yet).
DOCUMENTS_DIR = UPLOAD_DIR / "files"
METADATA_DIR = UPLOAD_DIR / "metadata"
IMAGES_DIR = UPLOAD_DIR / "images"
TEMP_DIR = UPLOAD_DIR / "temp"

# Enforced by app.services.document_storage on upload. Parsing strategies for
# each type live in app.services.parsers (legacy .doc/.xls/.ppt raise clearly).
ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".xlsx",
    ".pptx",
    ".rtf",
}

# Embedding configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "voyage": "voyage-4-lite",
    "cohere": "embed-v4.0",
}

# Per-provider model overrides in .env (preferred).
# EMBEDDING_MODEL is kept as a backward-compatible alias for OpenAI only.
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv(
    "EMBEDDING_MODEL"
)
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL")

_PROVIDER_EMBEDDING_MODEL_ENV = {
    "openai": OPENAI_EMBEDDING_MODEL,
    "voyage": VOYAGE_EMBEDDING_MODEL,
    "cohere": COHERE_EMBEDDING_MODEL,
}

# Resolved default for the configured EMBEDDING_PROVIDER (legacy export).
EMBEDDING_MODEL = (
    _PROVIDER_EMBEDDING_MODEL_ENV.get(EMBEDDING_PROVIDER)
    or DEFAULT_EMBEDDING_MODELS.get(
        EMBEDDING_PROVIDER,
        DEFAULT_EMBEDDING_MODELS["openai"],
    )
)

# API keys for embedding providers. Validated when a provider is instantiated.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def default_embedding_model(provider: str) -> str:
    """Return the default model name for an embedding provider."""
    return DEFAULT_EMBEDDING_MODELS.get(provider, DEFAULT_EMBEDDING_MODELS["openai"])


def resolve_embedding_model(provider: str) -> str:
    """Return the provider-specific model override or that provider's default."""
    configured = _PROVIDER_EMBEDDING_MODEL_ENV.get(provider)
    if configured:
        return configured
    return default_embedding_model(provider)

# Persistent Chroma storage for app.vector_store.chroma_vector_store.
_chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_PERSIST_DIR = (
    Path(_chroma_persist_dir)
    if _chroma_persist_dir
    else BACKEND_ROOT / "chroma_data"
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")

# Temporary flags for structured pipeline logging during development.
# Set RAG_DEBUG=true or CHUNKING_DEBUG=true in .env.
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() in ("true", "1", "yes")
CHUNKING_DEBUG = os.getenv("CHUNKING_DEBUG", "false").lower() in ("true", "1", "yes")

# Default chunk size for document splitting. Table blocks use 2x this limit.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = CHUNK_SIZE // 5