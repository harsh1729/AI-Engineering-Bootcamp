import uuid

from sqlalchemy import or_, select

from app.database.models.stored_document import Document
from app.database.repositories.base import BaseRepository


class DocumentRepository(BaseRepository):
    async def create(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        chunking_strategy: str,
        embedding_provider: str,
        vector_db: str,
        chunk_count: int = 0,
        content_type: str = "text",
        user_id: uuid.UUID | None = None,
        guest_id: str | None = None,
    ) -> Document:
        if user_id is None and guest_id is None:
            raise ValueError("Either user_id or guest_id must be provided.")

        document = Document(
            id=document_id,
            user_id=user_id,
            guest_id=guest_id,
            filename=filename,
            chunking_strategy=chunking_strategy,
            embedding_provider=embedding_provider,
            vector_db=vector_db,
            chunk_count=chunk_count,
            content_type=content_type,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list_by_guest(self, guest_id: str) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.guest_id == guest_id)
            .order_by(Document.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user(self, user_id: uuid.UUID) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user_and_guest(
        self,
        user_id: uuid.UUID,
        guest_id: str,
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(or_(Document.user_id == user_id, Document.guest_id == guest_id))
            .order_by(Document.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    def can_access(
        document: Document,
        *,
        user_id: uuid.UUID | None,
        guest_id: str | None,
    ) -> bool:
        if user_id is not None:
            if document.user_id == user_id:
                return True
            return guest_id is not None and document.guest_id == guest_id

        return guest_id is not None and document.guest_id == guest_id
