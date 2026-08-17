import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import admin, auth, chat, chats, documents, health

# Without this, the root logger defaults to WARNING and any logger.info(...)
# call anywhere in the app (e.g. chat.py's document_ids logging) is silently
# dropped.
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(chats.router)
app.include_router(documents.router)
