"""Tests for the pure pieces of topic-model selection.

Deriving the HDBSCAN min_topic_size grid from the corpus size, filtering grid
results to the desired topic-count band (topic_model.desired_topics), scoring a
model's topic balance, and ranking the candidates. The exhaustive grid itself is
exercised by the integration tests.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.topic import (derive_min_topic_size_range, filter_to_topic_band,
                               rank_by_distance, topic_entropy)
from pysyrev.core.config import TopicModelConfig


class TestDeriveMinTopicSizeRange:

    def test_maps_band_to_inverse_cluster_size(self):
        # min_cluster_size ~= N / nb_topics, so [t_min, t_max] -> [N/t_max, N/t_min].
        rng, step = derive_min_topic_size_range(200, [5, 15])
        assert rng == [13, 40]          # round(200/15), round(200/5)
        assert step >= 1

    def test_floor_at_two(self):
        # A tiny corpus with a large max-topics would give <2; clamp to 2.
        rng, _ = derive_min_topic_size_range(10, [5, 20])
        assert rng[0] >= 2

    def test_exact_target_gives_singleton_range(self):
        rng, step = derive_min_topic_size_range(200, [10, 10])
        assert rng == [20, 20]
        assert step == 1

    def test_step_is_positive(self):
        _, step = derive_min_topic_size_range(200, [8, 12])
        assert step >= 1


def _results(nb_topics):
    """A minimal ranked results frame with a nb_topics column."""
    return pd.DataFrame({
        "nb_topics": nb_topics,
        "distance":  [i * 0.1 for i in range(len(nb_topics))],
    })


class TestFilterToTopicBand:

    def test_keeps_only_in_band(self):
        out = filter_to_topic_band(_results([3, 8, 12, 26, 40]), [5, 15])
        assert out["nb_topics"].tolist() == [8, 12]

    def test_inclusive_bounds(self):
        out = filter_to_topic_band(_results([5, 10, 15]), [5, 15])
        assert out["nb_topics"].tolist() == [5, 10, 15]

    def test_no_range_is_passthrough(self):
        res = _results([3, 26])
        assert filter_to_topic_band(res, None) is res

    def test_empty_band_falls_back_to_all(self, capsys):
        res = _results([3, 26, 40])
        out = filter_to_topic_band(res, [8, 12])
        # Nothing in [8, 12] -> fall back to the full ranking, with a warning.
        assert out["nb_topics"].tolist() == [3, 26, 40]
        assert "no model produced a topic count" in capsys.readouterr().out


class TestDesiredTopicsConfigValidation:

    def _cfg(self, **kw):
        return TopicModelConfig(export={"export_dir": "/tmp/x"}, **kw)

    def test_valid_band(self):
        assert self._cfg(desired_topics=[5, 15]).desired_topics == [5, 15]

    def test_default_band(self):
        # desired_topics is the sole user knob for granularity — it must default.
        assert self._cfg().desired_topics == [5, 15]

    @pytest.mark.parametrize("bad", [[5], [5, 4], [0, 10], [1, 2, 3]])
    def test_invalid_bands_raise(self, bad):
        with pytest.raises(ValueError, match="desired_topics"):
            self._cfg(desired_topics=bad)


# =============================================================================
# rank_by_distance — the ranking is relative to the models still in contention
# =============================================================================

def _grid():
    """Four models: three inside a [5, 10] band, one degenerate 2-topic fit.

    The three candidates trade off — more topics buys entropy and costs
    coherence, coverage and diversity — so which one wins says which metric the
    Chebyshev max is reading. The out-of-band model is middling everywhere
    except entropy, where it sits far below the rest.
    """
    return pd.DataFrame({
        "nb_topics":   [5,     6,     8,     2],
        "entropy":     [1.00,  1.05,  1.10,  0.10],
        "u_mass":      [-0.10, -0.11, -0.12, -0.115],
        "diversity":   [0.90,  0.89,  0.88,  0.885],
        "nb_outliers": [10,    12,    14,    13],
    })


class TestRankByDistance:

    def test_adds_a_distance_column_sorted_best_first(self):
        ranked = rank_by_distance(_grid(), "chebyshev", 100, "u_mass")
        assert "distance" in ranked.columns
        assert ranked["distance"].is_monotonic_increasing

    def test_distance_sits_just_before_the_model_column(self):
        frame = _grid().assign(model=["a", "b", "c", "d"])
        ranked = rank_by_distance(frame, "chebyshev", 100, "u_mass")
        assert list(ranked.columns)[-2:] == ["distance", "model"]

    def test_a_lone_candidate_discriminates_nothing(self):
        ranked = rank_by_distance(_grid().head(1), "chebyshev", 100, "u_mass")
        assert ranked["distance"].tolist() == [0.0]

    def test_ranking_the_candidates_alone_picks_a_different_model(self):
        # Every metric is min-max normalized, so the models present set the scale.
        # Filtered first, the three candidates spread across the entropy scale and
        # the middle model wins on the trade-off.
        band = [5, 10]
        ranked = rank_by_distance(filter_to_topic_band(_grid(), band),
                                  "chebyshev", 100, "u_mass")
        assert ranked.iloc[0]["nb_topics"] == 6

        # Ranked over the whole grid and filtered afterwards, the out-of-band
        # model has stretched the entropy scale until entropy no longer separates
        # the candidates, and the smallest topic count takes the top slot — the
        # symptom this ordering was introduced to fix.
        stale = filter_to_topic_band(
            rank_by_distance(_grid(), "chebyshev", 100, "u_mass"), band)
        assert stale.iloc[0]["nb_topics"] == 5


# =============================================================================
# topic_entropy — balance of the topics, not a second count of the outliers
# =============================================================================

class TestTopicEntropy:

    def test_uniform_topics_reach_log_k(self):
        assert topic_entropy([50, 50, 50, 50]) == pytest.approx(math.log(4))

    def test_a_single_topic_carries_no_information(self):
        assert topic_entropy([700]) == 0.0

    def test_lopsided_topics_score_below_balanced_ones(self):
        assert topic_entropy([190, 10, 10]) < topic_entropy([70, 70, 70])

    def test_it_is_the_entropy_over_the_clustered_documents(self):
        sizes = [185, 173, 140, 111, 104]          # a real 5-topic fit
        p = np.asarray(sizes) / sum(sizes)
        assert topic_entropy(sizes) == pytest.approx(float(-(p * np.log(p)).sum()))

    def test_only_the_proportions_count(self):
        # The corpus size no longer enters: how many documents HDBSCAN left
        # unclustered is the coverage criterion's business, not this one.
        # Dividing by the whole corpus made this metric ~collinear with coverage.
        assert topic_entropy([100, 80, 60]) == pytest.approx(topic_entropy([200, 160, 120]))

    def test_no_topic_at_all(self):
        assert topic_entropy([]) == 0.0
        assert topic_entropy([0, 0]) == 0.0
