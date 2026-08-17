import uuid

from app.database.models.stored_document import Document
from app.database.repositories.document_repository import DocumentRepository
from app.models.rag_config import (
    ChunkingStrategy,
    EmbeddingProviderType,
    RagOptions,
    VectorStoreType,
)


class MixedRagConfigError(Exception):
    """Raised when attached documents were indexed with different RAG settings."""


class InvalidDocumentRagConfigError(Exception):
    """Raised when stored document metadata cannot be mapped to RAG options."""


def _rag_config_key(document: Document) -> tuple[str, str, str]:
    return (
        document.chunking_strategy,
        document.embedding_provider,
        document.vector_db,
    )


class RagOptionsResolverService:
    """Resolve retrieval pipeline settings from persisted document metadata."""

    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    async def resolve_rag_options_for_documents(
        self,
        document_ids: list[str],
    ) -> RagOptions:
        if not document_ids:
            raise ValueError("document_ids must not be empty.")

        documents: list[Document] = []
        for raw_id in document_ids:
            try:
                document_id = uuid.UUID(raw_id)
            except ValueError as exc:
                raise InvalidDocumentRagConfigError(
                    f"Document '{raw_id}' was not found."
                ) from exc

            document = await self._document_repository.get_by_id(document_id)
            if document is None:
                raise InvalidDocumentRagConfigError(
                    f"Document '{raw_id}' was not found."
                )
            documents.append(document)

        config_keys = {_rag_config_key(document) for document in documents}
        if len(config_keys) > 1:
            raise MixedRagConfigError(
                "Attached documents use different indexing settings. "
                "Remove documents or re-upload with matching RAG options."
            )

        return self._to_rag_options(documents[0])

    @staticmethod
    def _to_rag_options(document: Document) -> RagOptions:
        try:
            return RagOptions(
                chunking_strategy=ChunkingStrategy(document.chunking_strategy),
                embedding_provider=EmbeddingProviderType(document.embedding_provider),
                vector_store=VectorStoreType(document.vector_db),
            )
        except ValueError as exc:
            raise InvalidDocumentRagConfigError(
                f"Document '{document.id}' has invalid indexing configuration."
            ) from exc
