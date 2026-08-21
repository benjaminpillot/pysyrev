"""Tests for the desired-topics band feature (topic_model.desired_topics).

Covers the two pure pieces added for Option 1: deriving the HDBSCAN
min_topic_size grid from the corpus size, and filtering grid results to the
desired topic-count band. The exhaustive grid itself is exercised by the
integration tests.
"""

import pandas as pd
import pytest

from pysyrev.core.topic import derive_min_topic_size_range, filter_to_topic_band
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
