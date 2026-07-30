from app.retrieval.base_retriever import BaseRetriever
from app.retrieval.retrieval_models import RetrievalRequest, RetrievalResponse


class RetrievalService:
    """Public entry point for retrieving relevant document chunks."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return chunks relevant to `request` using the configured retriever."""
        return self._retriever.retrieve(request)
