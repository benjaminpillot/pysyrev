"""The desired topic band is a promise the search has to keep.

``desired_topics`` maps to a ``min_topic_size`` grid through a rule of thumb that
reads HDBSCAN's ``min_cluster_size`` as an average topic size. It is a floor, and
outlier documents join no topic at all, so the grid can land entirely above the
band: every model out of band, and no step through that grid able to reach it.

These tests drive ``topic_modeling`` against a corpus that splits at a known rate
(``nb_topics ~= K / min_topic_size``), so a grid aimed too high is guaranteed to
miss and the run has to notice, re-aim from what it measured, and sweep again.
The BERTopic fit is faked — what is under test is the search, not the clustering.
"""

import numpy as np
import pandas as pd
import pytest

from pysyrev.core import topic as topic_mod
from pysyrev.core.topic import topic_modeling

_N_DOCS = 60
_K = 400.0          # the corpus splits into ~K / min_topic_size topics
_OUTLIERS = 12


def _simulated_nb_topics(min_topic_size, min_samples):
    """The law the fake corpus obeys: inverse in the size, damped by min_samples."""
    n = _K / min_topic_size
    n -= min_samples / 10.0          # min_samples only ever takes the count down
    return max(1, int(round(n)))


class _FakeModel:
    """Just enough BERTopic surface for topic_modeling's export stage."""

    def __init__(self, nb_topics):
        self.nb_topics = nb_topics
        self.topics_ = [i % nb_topics for i in range(_N_DOCS)]
        self.c_tf_idf_ = None
        self.topic_representations_ = {t: [("w", 1.0)] for t in range(-1, nb_topics)}

    def approximate_distribution(self, documents, **kwargs):
        return np.zeros((len(documents), self.nb_topics)), None

    def get_topic_info(self):
        return pd.DataFrame({"Topic": list(range(-1, self.nb_topics)),
                             "Count": [_OUTLIERS] + [1] * self.nb_topics})

    def _extract_representative_docs(self, **kwargs):
        ids = [np.array([0, 1, 2]) for _ in range(self.nb_topics + 1)]
        return None, None, None, ids


class _FakeUmap:
    """Records every reduction actually computed, to prove they are reused."""

    def __init__(self):
        self.calls = []

    def __call__(self, n_neighbors, n_components, embeddings):
        self.calls.append((n_neighbors, n_components))
        return embeddings


class _FakeBertopic:
    def __init__(self):
        self.umap_model = _FakeUmap()

    def __call__(self, min_topic_size, min_samples):
        return {"min_topic_size": min_topic_size, "min_samples": min_samples}


@pytest.fixture
def fake_fit(monkeypatch):
    """Replace the BERTopic fit with the simulated law; record every pair tried."""
    tried = []

    def _fit(documents, reduced_embeddings, coherence_scorer=None,
             min_topic_size=None, min_samples=None, **kwargs):
        tried.append((min_topic_size, min_samples))
        nb = _simulated_nb_topics(min_topic_size, min_samples)
        sizes = {-1: _OUTLIERS}
        sizes.update({t: (_N_DOCS - _OUTLIERS) // nb for t in range(nb)})
        metrics = {"Coherence": -0.1 - nb / 1000, "Diversity": 0.9}
        return ([], None, sizes, _FakeModel(nb), {}, metrics)

    monkeypatch.setattr(topic_mod, "berteley_fit", _fit)
    return tried


def _run(tmp_path, *, min_topic_size_range, topic_range, min_samples_range=(2, 2),
         min_samples_step=1, topic_size_step=10, keep_n_results=5,
         show_progress=False):
    documents = [f"document number {i}" for i in range(_N_DOCS)]
    dataset = pd.DataFrame({"title": documents, "abstract": documents})
    bertopic_model = _FakeBertopic()
    topic_modeling(
        dataset, documents, bertopic_model,
        topic_distribution=_TopicDistribution(),
        embeddings=np.zeros((_N_DOCS, 4)),
        umap_n_neighbors=[5], umap_n_components=[3],
        min_topic_size_range=list(min_topic_size_range),
        min_samples_range=list(min_samples_range),
        topic_size_step=topic_size_step, min_samples_step=min_samples_step,
        export_to=str(tmp_path), nr_repr_docs=3, distance_name="euclidean",
        ranking_scorer="u_mass", purity_scorer="u_mass",
        keep_n_results=keep_n_results, show_progress=show_progress,
        surviving_indices=None, topic_range=topic_range,
    )
    results = pd.read_csv(tmp_path / "metrics" / "euclidean_results.csv")
    best = pd.read_csv(tmp_path / "metrics" / "euclidean_best_results.csv")
    return results, best, bertopic_model


class _TopicDistribution:
    window = 4
    stride = 1
    min_similarity = 0.1
    batch_size = 100


class TestTheBandIsReached:

    def test_a_grid_aimed_too_high_is_re_aimed_until_it_lands(self, tmp_path,
                                                              fake_fit, capsys):
        # [40, 90] can only yield 400/40 = 10 down to 400/90 = 4 topics, and the
        # step lands on 40, 50, ... — the derived grid a real corpus produces.
        results, best, _ = _run(tmp_path, min_topic_size_range=(40, 90),
                                topic_range=[15, 25])
        assert "Re-aiming" in capsys.readouterr().out
        # Every kept model is in band — the filter no longer falls back.
        assert best["nb_topics"].between(15, 25).all()

    def test_the_re_aimed_grid_is_swept_on_top_of_the_first(self, tmp_path, fake_fit):
        results, _, _ = _run(tmp_path, min_topic_size_range=(40, 90),
                             topic_range=[15, 25])
        swept = sorted({size for size, _ in fake_fit})
        assert max(swept) == 90                 # the first grid was swept in full
        assert min(swept) < 40                  # and the re-aimed one on top of it
        # What is reported is the in-band survivors, all from the re-aimed grid.
        assert topic_mod.min_topic_sizes_of(results).max() < 40

    def test_umap_reductions_are_computed_once_across_sweeps(self, tmp_path, fake_fit):
        # Re-aiming costs another HDBSCAN sweep; it must not cost another UMAP.
        _, _, bertopic_model = _run(tmp_path, min_topic_size_range=(40, 90),
                                    topic_range=[15, 25])
        assert bertopic_model.umap_model.calls == [(5, 3)]

    def test_a_grid_already_in_band_is_not_re_aimed(self, tmp_path, fake_fit, capsys):
        # 400/20 = 20 topics: the opening grid lands, so nothing to re-aim.
        _, best, _ = _run(tmp_path, min_topic_size_range=(20, 40),
                          topic_range=[10, 20])
        assert "Re-aiming" not in capsys.readouterr().out
        assert best["nb_topics"].between(10, 20).all()

    def test_without_a_band_the_grid_is_swept_once_as_given(self, tmp_path,
                                                            fake_fit, capsys):
        results, _, _ = _run(tmp_path, min_topic_size_range=(40, 90),
                             topic_range=None)
        assert "Re-aiming" not in capsys.readouterr().out
        assert sorted(topic_mod.min_topic_sizes_of(results).unique()) == [40, 50, 60,
                                                                         70, 80, 90]


class TestTheSearchGivesUpCleanly:

    def test_an_unreachable_band_stops_after_the_attempt_cap(self, tmp_path,
                                                             monkeypatch, capsys):
        # A corpus that always splits into 3 topics: no min_topic_size reaches the
        # band, so re-aiming cannot help and must not loop forever.
        def _fit(documents, reduced_embeddings, coherence_scorer=None,
                 min_topic_size=None, min_samples=None, **kwargs):
            sizes = {-1: _OUTLIERS, 0: 16, 1: 16, 2: 16}
            return ([], None, sizes, _FakeModel(3), {},
                    {"Coherence": -0.1, "Diversity": 0.9})

        monkeypatch.setattr(topic_mod, "berteley_fit", _fit)
        _, best, _ = _run(tmp_path, min_topic_size_range=(40, 90),
                          topic_range=[15, 25])
        out = capsys.readouterr().out
        assert out.count("Re-aiming") <= topic_mod._MAX_REAIM_ATTEMPTS
        assert "no model produced a topic count" in out
        assert len(best) > 0                    # the run still produced its output


class TestTheProgressBarPath:
    """Every other test runs silent, which is how a NameError in the reporting
    path survived a green suite: the bar is opened and closed by `sweep`, once
    per grid, and nothing outside it may touch that handle."""

    def test_a_run_with_the_bar_on_completes(self, tmp_path, fake_fit):
        _, best, _ = _run(tmp_path, min_topic_size_range=(20, 40),
                          topic_range=[10, 20], show_progress=True)
        assert len(best) > 0

    def test_a_re_aimed_run_with_the_bar_on_completes(self, tmp_path, fake_fit):
        # Two sweeps means two bars: the second must not inherit the first.
        _, best, _ = _run(tmp_path, min_topic_size_range=(40, 90),
                          topic_range=[15, 25], show_progress=True)
        assert best["nb_topics"].between(15, 25).all()


class TestUnusedMinSamplesAreReported:
    """min_samples cannot be derived, so a range set too wide is only visible
    after the fact — from which part of it the ranking actually selected."""

    def _report(self, selected_ms, fitted_ms, capsys):
        frame = lambda ms: pd.DataFrame({"hdbscan": [f"20_{m}" for m in ms]})
        topic_mod.report_unused_min_samples(frame(selected_ms), frame(fitted_ms))
        return capsys.readouterr().out

    def test_a_whole_unused_corner_is_named(self, capsys):
        out = self._report([2, 8, 18], list(range(2, 51, 2)), capsys)
        assert "min_samples above 18" in out
        assert "min_sample_range [2, 18]" in out

    def test_a_winner_near_the_ceiling_is_not_a_finding(self, capsys):
        # One step short of the top is the ranking doing its job, not waste.
        assert self._report([2, 48], list(range(2, 51, 2)), capsys) == ""

    def test_it_measures_against_what_was_fitted_not_what_was_asked_for(self, capsys):
        # The degenerate pairs are never fitted, so they must not be reported as
        # fits that won nothing.
        assert self._report([2, 8], [2, 4, 6, 8], capsys) == ""

    def test_a_grid_too_small_to_judge_stays_quiet(self, capsys):
        assert self._report([2], [2, 4], capsys) == ""


class TestDegenerateMinSamplesAreSkipped:

    def test_min_samples_above_min_topic_size_is_not_fitted(self, tmp_path, fake_fit):
        # min_samples > min_cluster_size makes the density threshold stricter than
        # the cluster it should keep: those fits can only lose topics.
        _run(tmp_path, min_topic_size_range=(20, 40), topic_range=[10, 20],
             min_samples_range=(2, 60), min_samples_step=10, topic_size_step=10)
        assert fake_fit                                    # something was fitted
        assert all(ms <= size for size, ms in fake_fit)

    def test_a_min_sample_range_entirely_above_the_grid_is_still_swept(
            self, tmp_path, fake_fit):
        # Nothing else to fall back on: the user asked for this corner explicitly.
        _run(tmp_path, min_topic_size_range=(20, 20), topic_range=None,
             min_samples_range=(50, 50), min_samples_step=1, topic_size_step=1)
        assert fake_fit == [(20, 50)]

    def test_a_size_below_the_whole_min_samples_grid_is_still_fitted(
            self, tmp_path, fake_fit):
        # Sizes 5 and 15 sit below every min_samples on offer while 25 and 35 do
        # not. Dropping their degenerate pairs must not drop the sizes: they are
        # the small ones, the ones a re-aim reaches for when it needs more topics.
        _run(tmp_path, min_topic_size_range=(5, 35), topic_range=None,
             min_samples_range=(20, 30), min_samples_step=10, topic_size_step=10)
        fitted = {size: sorted(ms for s, ms in fake_fit if s == size)
                  for size in (5, 15, 25, 35)}
        assert fitted[5] == [20]           # fitted once, at the least conservative
        assert fitted[15] == [20]
        assert fitted[25] == [20]          # 30 > 25, dropped as degenerate
        assert fitted[35] == [20, 30]      # both usable
