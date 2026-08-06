import pytest

from app.embeddings.embedding_exceptions import EmbeddingProviderError
from app.embeddings.embedding_provider_utils import (
    build_batch_from_indexed_items,
    empty_batch_response,
)


class TestEmbeddingProviderUtils:
    def test_empty_batch_response_has_zero_usage(self) -> None:
        batch = empty_batch_response()

        assert batch.embeddings == []
        assert batch.usage.prompt_tokens == 0
        assert batch.usage.total_tokens == 0

    def test_build_batch_from_indexed_items_preserves_input_order(self) -> None:
        batch = build_batch_from_indexed_items(
            [(2, [3.0]), (0, [1.0]), (1, [2.0])],
            expected_count=3,
            model="test-model",
            prompt_tokens=10,
            total_tokens=10,
        )

        assert [vector.embedding for vector in batch.embeddings] == [
            [1.0],
            [2.0],
            [3.0],
        ]
        assert all(vector.model == "test-model" for vector in batch.embeddings)

    def test_build_batch_from_indexed_items_raises_on_missing_index(self) -> None:
        with pytest.raises(
            EmbeddingProviderError,
            match=r"Missing embedding response indices: \[1\]",
        ):
            build_batch_from_indexed_items(
                [(0, [1.0])],
                expected_count=2,
                model="test-model",
                prompt_tokens=1,
                total_tokens=1,
            )
