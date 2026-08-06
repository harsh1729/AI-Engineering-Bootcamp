import importlib

import pytest


class TestResolveEmbeddingModel:
    def test_openai_uses_embedding_model_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
        config = importlib.reload(importlib.import_module("app.config"))

        assert config.resolve_embedding_model("openai") == "text-embedding-3-large"

    def test_voyage_ignores_openai_embedding_model_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.delenv("VOYAGE_EMBEDDING_MODEL", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
        config = importlib.reload(importlib.import_module("app.config"))

        assert config.resolve_embedding_model("voyage") == "voyage-4-lite"

    def test_provider_specific_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "voyage-3-large")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
        config = importlib.reload(importlib.import_module("app.config"))

        assert config.resolve_embedding_model("voyage") == "voyage-3-large"
        assert config.resolve_embedding_model("openai") == "text-embedding-3-small"
