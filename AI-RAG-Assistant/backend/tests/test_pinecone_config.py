import pytest

from app.config import (
    PINECONE_INDEX_BY_PROVIDER,
    pinecone_indexes_catalog,
    resolve_pinecone_dimension,
    resolve_pinecone_index,
)


class TestPineconeConfigResolver:
    def test_resolve_index_for_each_provider(self) -> None:
        for provider in ("openai", "voyage", "cohere"):
            assert resolve_pinecone_index(provider) == PINECONE_INDEX_BY_PROVIDER[provider]
            assert resolve_pinecone_dimension(provider) > 0

    def test_resolve_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="No Pinecone index configured"):
            resolve_pinecone_index("unknown")

    def test_pinecone_indexes_catalog_includes_all_providers(self) -> None:
        catalog = pinecone_indexes_catalog()

        assert set(catalog) == {"openai", "voyage", "cohere"}
        assert catalog["openai"]["dimension"] == resolve_pinecone_dimension("openai")
        assert catalog["voyage"]["index_name"] == resolve_pinecone_index("voyage")
