import pytest

from app.config import (
    QDRANT_COLLECTION_BY_PROVIDER,
    qdrant_collections_catalog,
    resolve_qdrant_collection,
    resolve_qdrant_dimension,
)


class TestQdrantConfigResolver:
    def test_resolve_collection_for_each_provider(self) -> None:
        for provider in ("openai", "voyage", "cohere"):
            assert resolve_qdrant_collection(provider) == QDRANT_COLLECTION_BY_PROVIDER[provider]
            assert resolve_qdrant_dimension(provider) > 0

    def test_resolve_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="No Qdrant collection configured"):
            resolve_qdrant_collection("unknown")

    def test_qdrant_collections_catalog_includes_all_providers(self) -> None:
        catalog = qdrant_collections_catalog()

        assert set(catalog) == {"openai", "voyage", "cohere"}
        assert catalog["openai"]["dimension"] == resolve_qdrant_dimension("openai")
        assert catalog["cohere"]["collection_name"] == resolve_qdrant_collection("cohere")
