import chromadb
from chromadb.api import ClientAPI
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api.types import QueryResult

from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_store_models import VectorQueryResult, VectorStoreRecord

_COSINE_CONFIGURATION = CreateCollectionConfiguration(hnsw={"space": "cosine"})


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by a persistent local Chroma collection."""

    def __init__(
        self,
        client: ClientAPI | None = None,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> None:
        if client is None:
            directory = persist_directory or str(CHROMA_PERSIST_DIR)
            client = chromadb.PersistentClient(path=directory)

        self._client = client
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            configuration=_COSINE_CONFIGURATION,
        )

    def upsert(self, records: list[VectorStoreRecord]) -> None:
        if not records:
            return

        self._collection.upsert(
            ids=[record.chunk_id for record in records],
            documents=[record.text for record in records],
            embeddings=[record.embedding.embedding for record in records],
            metadatas=[
                {
                    "document_id": record.document_id,
                    "chunk_index": record.chunk_index,
                    "source_filename": record.source_filename,
                }
                for record in records
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorQueryResult]:
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if document_ids:
            query_kwargs["where"] = {"document_id": {"$in": document_ids}}

        response = self._collection.query(**query_kwargs)
        return self._map_query_response(response)

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def _map_query_response(self, response: QueryResult) -> list[VectorQueryResult]:
        ids = response["ids"]
        documents = response["documents"]
        metadatas = response["metadatas"]
        distances = response["distances"]

        if not ids or not ids[0]:
            return []

        if documents is None or metadatas is None or distances is None:
            raise ValueError(
                "Chroma query response is missing documents, metadata, or distances"
            )

        return [
            VectorQueryResult(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                chunk_index=metadata["chunk_index"],
                text=document,
                source_filename=metadata["source_filename"],
                distance=distance,
            )
            for chunk_id, document, metadata, distance in zip(
                ids[0],
                documents[0],
                metadatas[0],
                distances[0],
                strict=True,
            )
        ]
