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

# Embedding model used by app.embeddings.openai_embedding_provider.
# Override via EMBEDDING_MODEL in the environment.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI API key for embedding requests. Validated when OpenAIEmbeddingProvider
# is instantiated without an injected client.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Persistent Chroma storage for app.vector_store.chroma_vector_store.
_chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_PERSIST_DIR = (
    Path(_chroma_persist_dir)
    if _chroma_persist_dir
    else BACKEND_ROOT / "chroma_data"
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")

# Temporary flag for structured RAG pipeline logging. Set RAG_DEBUG=true in .env.
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() in ("true", "1", "yes")