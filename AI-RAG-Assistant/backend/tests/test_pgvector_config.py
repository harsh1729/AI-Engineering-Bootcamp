import pytest

from app.config import (
    PGVECTOR_TABLE_BY_PROVIDER,
    pgvector_tables_catalog,
    resolve_pgvector_dimension,
    resolve_pgvector_table,
    sync_database_url,
)


class TestPgVectorConfigResolver:
    def test_resolve_table_for_each_provider(self) -> None:
        for provider in ("openai", "voyage", "cohere"):
            assert resolve_pgvector_table(provider) == PGVECTOR_TABLE_BY_PROVIDER[provider]
            assert resolve_pgvector_dimension(provider) > 0

    def test_resolve_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="No pgvector table configured"):
            resolve_pgvector_table("unknown")

    def test_pgvector_tables_catalog_includes_all_providers(self) -> None:
        catalog = pgvector_tables_catalog()

        assert set(catalog) == {"openai", "voyage", "cohere"}
        assert catalog["openai"]["dimension"] == resolve_pgvector_dimension("openai")
        assert catalog["voyage"]["table_name"] == resolve_pgvector_table("voyage")

    def test_sync_database_url_converts_asyncpg_driver(self) -> None:
        assert sync_database_url().startswith("postgresql+psycopg://")
