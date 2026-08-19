from pathlib import Path
import os

from dotenv import load_dotenv

# Resolved relative to this file (not the process CWD) so uploads always land
# in the same place regardless of where uvicorn is launched from.
BACKEND_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(BACKEND_ROOT / ".env")

# Maximum number of user messages a single guest may send during the demo.
# Enforced by app.services.usage_tracker.
MAX_DEMO_INTERACTIONS = 50

# Maximum messages for a logged-in user (shared across all LLM providers).
LOGGED_IN_USER_INTERACTIONS = 10 * MAX_DEMO_INTERACTIONS

# Shown when a guest exhausts MAX_DEMO_INTERACTIONS (shared across all LLM providers).
GUEST_USAGE_LIMIT_MESSAGE = (
    "Guest user limit reached. Register for more messages."
)

# Shown when a logged-in user exhausts LOGGED_IN_USER_INTERACTIONS.
LOGGED_IN_USER_USAGE_LIMIT_MESSAGE = (
    "Account message limit reached. Please contact support."
)

ACCOUNT_PENDING_APPROVAL_MESSAGE = (
    "Your account is pending approval. Please try again later."
)

ACCOUNT_DISABLED_MESSAGE = (
    "Your account has been disabled. Please contact an administrator."
)

REGISTRATION_PENDING_MESSAGE = (
    "Registration successful. Your account is pending admin approval."
)

# JWT auth (set JWT_SECRET_KEY in production).
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "dev-only-change-me-in-production",
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))

# Maximum size of a single uploaded document. Enforced by the upload endpoint
# before the file is written to disk (and mirrored client-side for UX).
MAX_DOCUMENT_SIZE_MB = 15
MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024

# Maximum size of a single uploaded image (OCR path). Documents use MAX_DOCUMENT_SIZE_MB.
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

UPLOAD_DIR = BACKEND_ROOT / "uploads"

# uploads/ is a root with purpose-specific subfolders:
# - files/  text document uploads (PDF, DOCX, etc.).
# - images/ OCR-indexed image uploads (JPG, PNG, WEBP).
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

# Image uploads (OCR via app.services.parsers.image_parser).
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

ALLOWED_UPLOAD_EXTENSIONS = ALLOWED_DOCUMENT_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

# Document vs image content types persisted on the documents table.
DOCUMENT_CONTENT_TYPE_TEXT = "text"
DOCUMENT_CONTENT_TYPE_IMAGE = "image"

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

# Pinecone vector store (app.vector_store.pinecone_vector_store).
# One index per embedding provider (dimensions must match the provider's vectors).
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

DEFAULT_EMBEDDING_DIMENSIONS = {
    "openai": 1536,
    "voyage": 1024,
    "cohere": 1536,
}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


_legacy_pinecone_index = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_OPENAI = (
    os.getenv("PINECONE_INDEX_OPENAI") or _legacy_pinecone_index or "ai-rag-openai-1536"
)
PINECONE_INDEX_VOYAGE = os.getenv("PINECONE_INDEX_VOYAGE", "ai-rag-voyage-1024")
PINECONE_INDEX_COHERE = os.getenv("PINECONE_INDEX_COHERE", "ai-rag-cohere-1536")

# Backward-compatible alias for OpenAI index name.
PINECONE_INDEX_NAME = PINECONE_INDEX_OPENAI

PINECONE_DIMENSION_OPENAI = _int_env("PINECONE_DIMENSION_OPENAI", 1536)
PINECONE_DIMENSION_VOYAGE = _int_env("PINECONE_DIMENSION_VOYAGE", 1024)
PINECONE_DIMENSION_COHERE = _int_env("PINECONE_DIMENSION_COHERE", 1536)

PINECONE_INDEX_BY_PROVIDER: dict[str, str] = {
    "openai": PINECONE_INDEX_OPENAI,
    "voyage": PINECONE_INDEX_VOYAGE,
    "cohere": PINECONE_INDEX_COHERE,
}

PINECONE_DIMENSION_BY_PROVIDER: dict[str, int] = {
    "openai": PINECONE_DIMENSION_OPENAI,
    "voyage": PINECONE_DIMENSION_VOYAGE,
    "cohere": PINECONE_DIMENSION_COHERE,
}


def resolve_pinecone_index(embedding_provider: str) -> str:
    """Return the Pinecone index name for an embedding provider."""
    index = PINECONE_INDEX_BY_PROVIDER.get(embedding_provider)
    if not index:
        raise ValueError(
            f"No Pinecone index configured for embedding provider '{embedding_provider}'."
        )
    return index


def resolve_pinecone_dimension(embedding_provider: str) -> int:
    """Return the expected vector dimension for an embedding provider's Pinecone index."""
    dimension = PINECONE_DIMENSION_BY_PROVIDER.get(embedding_provider)
    if dimension is None:
        raise ValueError(
            "No Pinecone dimension configured for embedding provider "
            f"'{embedding_provider}'."
        )
    return dimension


def pinecone_indexes_catalog() -> dict[str, dict[str, int | str]]:
    """Return embedding_provider -> {index_name, dimension} for API consumers."""
    return {
        provider: {
            "index_name": PINECONE_INDEX_BY_PROVIDER[provider],
            "dimension": PINECONE_DIMENSION_BY_PROVIDER[provider],
        }
        for provider in PINECONE_INDEX_BY_PROVIDER
    }

# pgvector vector store (app.vector_store.pgvector_vector_store).
# One table per embedding provider (dimensions must match the provider's vectors).
PGVECTOR_ENABLED = os.getenv("PGVECTOR_ENABLED", "true").lower() in ("true", "1", "yes")

PGVECTOR_TABLE_OPENAI = os.getenv("PGVECTOR_TABLE_OPENAI", "doc_pg_embeddings_openai")
PGVECTOR_TABLE_VOYAGE = os.getenv("PGVECTOR_TABLE_VOYAGE", "doc_pg_embeddings_voyage")
PGVECTOR_TABLE_COHERE = os.getenv("PGVECTOR_TABLE_COHERE", "doc_pg_embeddings_cohere")

PGVECTOR_TABLE_BY_PROVIDER: dict[str, str] = {
    "openai": PGVECTOR_TABLE_OPENAI,
    "voyage": PGVECTOR_TABLE_VOYAGE,
    "cohere": PGVECTOR_TABLE_COHERE,
}


def resolve_pgvector_table(embedding_provider: str) -> str:
    """Return the pgvector table name for an embedding provider."""
    table = PGVECTOR_TABLE_BY_PROVIDER.get(embedding_provider)
    if not table:
        raise ValueError(
            f"No pgvector table configured for embedding provider '{embedding_provider}'."
        )
    return table


def resolve_pgvector_dimension(embedding_provider: str) -> int:
    """Return the expected vector dimension for an embedding provider's pgvector table."""
    return resolve_pinecone_dimension(embedding_provider)


def pgvector_tables_catalog() -> dict[str, dict[str, int | str]]:
    """Return embedding_provider -> {table_name, dimension} for API consumers."""
    return {
        provider: {
            "table_name": PGVECTOR_TABLE_BY_PROVIDER[provider],
            "dimension": resolve_pgvector_dimension(provider),
        }
        for provider in PGVECTOR_TABLE_BY_PROVIDER
    }

# Qdrant vector store (app.vector_store.qdrant_vector_store).
# One collection per embedding provider (dimensions must match the provider's vectors).
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_COLLECTION_OPENAI = os.getenv("QDRANT_COLLECTION_OPENAI", "ai-rag-openai-1536")
QDRANT_COLLECTION_VOYAGE = os.getenv("QDRANT_COLLECTION_VOYAGE", "ai-rag-voyage-1024")
QDRANT_COLLECTION_COHERE = os.getenv("QDRANT_COLLECTION_COHERE", "ai-rag-cohere-1536")

QDRANT_DIMENSION_OPENAI = _int_env("QDRANT_DIMENSION_OPENAI", 1536)
QDRANT_DIMENSION_VOYAGE = _int_env("QDRANT_DIMENSION_VOYAGE", 1024)
QDRANT_DIMENSION_COHERE = _int_env("QDRANT_DIMENSION_COHERE", 1536)

QDRANT_COLLECTION_BY_PROVIDER: dict[str, str] = {
    "openai": QDRANT_COLLECTION_OPENAI,
    "voyage": QDRANT_COLLECTION_VOYAGE,
    "cohere": QDRANT_COLLECTION_COHERE,
}

QDRANT_DIMENSION_BY_PROVIDER: dict[str, int] = {
    "openai": QDRANT_DIMENSION_OPENAI,
    "voyage": QDRANT_DIMENSION_VOYAGE,
    "cohere": QDRANT_DIMENSION_COHERE,
}


def resolve_qdrant_collection(embedding_provider: str) -> str:
    """Return the Qdrant collection name for an embedding provider."""
    collection = QDRANT_COLLECTION_BY_PROVIDER.get(embedding_provider)
    if not collection:
        raise ValueError(
            f"No Qdrant collection configured for embedding provider '{embedding_provider}'."
        )
    return collection


def resolve_qdrant_dimension(embedding_provider: str) -> int:
    """Return the expected vector dimension for an embedding provider's Qdrant collection."""
    dimension = QDRANT_DIMENSION_BY_PROVIDER.get(embedding_provider)
    if dimension is None:
        raise ValueError(
            "No Qdrant dimension configured for embedding provider "
            f"'{embedding_provider}'."
        )
    return dimension


def qdrant_collections_catalog() -> dict[str, dict[str, int | str]]:
    """Return embedding_provider -> {collection_name, dimension} for API consumers."""
    return {
        provider: {
            "collection_name": QDRANT_COLLECTION_BY_PROVIDER[provider],
            "dimension": QDRANT_DIMENSION_BY_PROVIDER[provider],
        }
        for provider in QDRANT_COLLECTION_BY_PROVIDER
    }

# Temporary flags for structured pipeline logging during development.
# Set RAG_DEBUG=true or CHUNKING_DEBUG=true in .env.
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() in ("true", "1", "yes")
CHUNKING_DEBUG = os.getenv("CHUNKING_DEBUG", "false").lower() in ("true", "1", "yes")

# Default chunk size for document splitting. Table blocks use 2x this limit.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = CHUNK_SIZE // 5

# PostgreSQL (async SQLAlchemy + asyncpg).
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/ai_rag_assistant",
)
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() in ("true", "1", "yes")


def sync_database_url() -> str:
    """Convert the async SQLAlchemy DATABASE_URL to a sync psycopg URL."""
    if DATABASE_URL.startswith("postgresql+asyncpg://"):
        return DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    return DATABASE_URL