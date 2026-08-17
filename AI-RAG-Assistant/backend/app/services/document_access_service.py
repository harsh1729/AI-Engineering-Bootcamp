import uuid

from app.database.models.stored_document import Document
from app.database.repositories.document_repository import DocumentRepository


class DocumentAccessDeniedError(Exception):
    """Raised when a caller tries to use a document they do not own."""


class DocumentAccessService:
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    async def list_documents(
        self,
        *,
        user_id: uuid.UUID | None,
        guest_id: str | None,
    ) -> list[Document]:
        if user_id is not None:
            if guest_id:
                return await self._document_repository.list_by_user_and_guest(
                    user_id,
                    guest_id,
                )
            return await self._document_repository.list_by_user(user_id)

        if not guest_id:
            raise ValueError("guest_id is required for unauthenticated requests.")

        return await self._document_repository.list_by_guest(guest_id)

    async def validate_document_ids(
        self,
        document_ids: list[str],
        *,
        user_id: uuid.UUID | None,
        guest_id: str | None,
    ) -> None:
        if not document_ids:
            return

        for raw_id in document_ids:
            try:
                document_id = uuid.UUID(raw_id)
            except ValueError as exc:
                raise DocumentAccessDeniedError(
                    f"Document '{raw_id}' was not found."
                ) from exc

            document = await self._document_repository.get_by_id(document_id)
            if document is None or not DocumentRepository.can_access(
                document,
                user_id=user_id,
                guest_id=guest_id,
            ):
                raise DocumentAccessDeniedError(
                    f"Document '{raw_id}' was not found."
                )
