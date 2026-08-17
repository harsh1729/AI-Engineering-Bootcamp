# AI RAG Assistant

A production-quality Retrieval-Augmented Generation (RAG) assistant, built incrementally as part of the AI Engineering Bootcamp.

## Dependency on LLM-SDK

This project does not talk to OpenAI, Claude, or Gemini directly. All LLM interactions (text generation, streaming, tool calling) go through the local `LLM-SDK` package.

See [`PROJECT_GUIDE.md`](./PROJECT_GUIDE.md) for the project vision, roadmap, and development rules.

## Structure

```
AI-RAG-Assistant/
├── backend/
│   ├── app/            # Application code (added as phases require it)
│   ├── tests/          # Automated tests
│   ├── logs/           # Runtime logs (gitignored)
│   ├── uploads/        # User-uploaded documents (gitignored)
│   └── requirements.txt
├── frontend/            # Frontend application (added when needed)
├── PROJECT_GUIDE.md
└── README.md
```

## Status

Project scaffold only. No application code, dependencies, or endpoints have been added yet — see `PROJECT_GUIDE.md` for the roadmap.

## PostgreSQL

The backend includes an async SQLAlchemy layer (`asyncpg`) for production-style persistence. This phase adds **schema and migrations only** — auth, chat persistence, and API wiring come later.

### Architecture notes

| Layer | Location | Purpose |
|-------|----------|---------|
| Database engine / sessions | `backend/app/database/` | Async engine, `AsyncSession`, FastAPI-ready `get_db_session()` |
| SQLAlchemy ORM | `backend/app/database/models/` | PostgreSQL tables (`User`, `Chat`, `Message`, `Document`) |
| Pydantic schemas | `backend/app/models/` | API request/response models (unchanged) |

Import ORM symbols from `app.database.models` or `app.database`.

**Repositories** live in `backend/app/database/repositories/` with FastAPI dependencies in `app/database/deps.py`:

```python
from app.database.deps import get_document_repository
from app.database.repositories import DocumentRepository
```

**Tables:** `users`, `chats`, `messages`, `documents`

- `User` → `Chat` → `Message` (cascade delete)
- `Document` stores indexed file metadata (no `user_id` yet — add when auth lands)

### Install PostgreSQL

**Option A — Docker (recommended for local dev)**

From the project root:

```bash
docker compose up -d
```

This starts PostgreSQL 16 on port `5432` with:

- user: `postgres`
- password: `password`
- database: `ai_rag_assistant`

**Option B — Local PostgreSQL**

Install PostgreSQL 16+, then create the database:

```sql
CREATE DATABASE ai_rag_assistant;
```

### Configure environment

Copy the example env file and adjust if needed:

```bash
cp backend/.env.example backend/.env
```

Required variable:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/ai_rag_assistant
```

Optional:

```env
DATABASE_ECHO=false
```

### Install Python dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run migrations

From `backend/` with your virtualenv active:

```bash
alembic upgrade head
```

Create a new migration after model changes:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

### Verify tables

Connect with `psql`:

```bash
psql postgresql://postgres:password@localhost:5432/ai_rag_assistant
```

List tables:

```sql
\dt
```

Expected tables: `users`, `chats`, `messages`, `documents`, `alembic_version`.

Quick row counts:

```sql
SELECT
  (SELECT COUNT(*) FROM users) AS users,
  (SELECT COUNT(*) FROM chats) AS chats,
  (SELECT COUNT(*) FROM messages) AS messages,
  (SELECT COUNT(*) FROM documents) AS documents;
```
