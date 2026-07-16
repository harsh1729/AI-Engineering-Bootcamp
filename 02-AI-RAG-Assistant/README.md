# 02 — AI RAG Assistant

A production-quality Retrieval-Augmented Generation (RAG) assistant, built incrementally as part of the AI Engineering Bootcamp.

## Dependency on 01-LLM-SDK

This project does not talk to OpenAI, Claude, or Gemini directly. All LLM interactions (text generation, streaming, tool calling) go through the local `01-LLM-SDK` package.

See [`PROJECT_GUIDE.md`](./PROJECT_GUIDE.md) for the project vision, roadmap, and development rules.

## Structure

```
02-AI-RAG-Assistant/
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
