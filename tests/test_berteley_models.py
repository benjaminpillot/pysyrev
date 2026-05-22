"""
Tests for pysyrev.core.berteley.models.

Fast tests: cover pure helpers (initialize_model n_gram_range logic,
    _calculate_topic_sizes) with no model downloads.
Integration tests (marked `integration`): run fit() + _calculate_metrics()
    with a real BERTopic stack — expect several minutes.
    Run with: pytest -m integration
"""

from unittest.mock import MagicMock

import pytest

from pysyrev.core.berteley.models import (
    _calculate_topic_sizes,
    initialize_model,
)


# =============================================================================
# _calculate_topic_sizes
# =============================================================================

class TestCalculateTopicSizes:
    def test_basic(self):
        topics = [0, 1, 0, 2, 1, -1]
        result = _calculate_topic_sizes(topics)
        assert result == {0: 2, 1: 2, 2: 1, -1: 1}

    def test_all_outliers(self):
        result = _calculate_topic_sizes([-1, -1, -1])
        assert result == {-1: 3}

    def test_single_topic(self):
        result = _calculate_topic_sizes([0, 0, 0])
        assert result == {0: 3}

    def test_empty(self):
        assert _calculate_topic_sizes([]) == {}


# =============================================================================
# initialize_model — n_gram_range resolution (no model download)
# =============================================================================

class TestInitializeModelNgramRange:
    """Pass a mock as embedding_model to bypass the string→model resolution
    and focus on testing the n_gram_range branch only."""

    def _call(self, n_gram_range):
        mock_model = MagicMock()
        _, resolved = initialize_model(mock_model, None, n_gram_range)
        return resolved

    def test_unigram_string(self):
        assert self._call("unigram") == (1, 1)

    def test_bigram_string(self):
        assert self._call("bigram") == (2, 2)

    def test_tuple_passthrough(self):
        assert self._call((1, 3)) == (1, 3)

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError):
            self._call("trigram")

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            self._call(42)

    def test_returns_original_model_unchanged(self):
        mock_model = MagicMock()
        returned_model, _ = initialize_model(mock_model, None, "bigram")
        assert returned_model is mock_model


# =============================================================================
# Integration — real BERTopic fit (slow, requires sentence-transformers + gensim)
# =============================================================================

@pytest.mark.integration
class TestFitIntegration:
    def test_fit_returns_expected_structure(self, tiny_corpus):
        from umap import UMAP
        from pysyrev.core.berteley.models import fit
        topics, probs, topic_sizes, topic_model, topic_words, metrics = fit(
            tiny_corpus,
            embedding_model="specter2",
            n_gram_range="unigram",
            coherence_scorer="u_mass",
            min_topic_size=2,
            umap_model=UMAP(n_neighbors=5, n_components=3, min_dist=0.0, random_state=42),
        )
        assert isinstance(topics, list)
        assert len(topics) == len(tiny_corpus)
        assert isinstance(topic_sizes, dict)
        assert "Coherence" in metrics
        assert "Diversity" in metrics

    def test_calculate_metrics_keys(self, tiny_corpus):
        from umap import UMAP
        from pysyrev.core.berteley.models import fit, _calculate_metrics
        _, _, _, topic_model, _, _ = fit(
            tiny_corpus,
            embedding_model="specter2",
            n_gram_range="unigram",
            coherence_scorer="u_mass",
            min_topic_size=2,
            umap_model=UMAP(n_neighbors=5, n_components=3, min_dist=0.0, random_state=42),
        )
        metrics = _calculate_metrics(tiny_corpus, topic_model, topic_model.topics_, "u_mass")
        assert set(metrics.keys()) == {"Coherence", "Diversity"}
        assert 0.0 <= metrics["Diversity"] <= 1.0
