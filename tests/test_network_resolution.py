"""Tests for the Leiden resolution sweep on bibliographic networks."""

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.networks.common import (
    resolution_grid, leiden_best_resolution, leiden_communities)
from pysyrev.core.networks.coupling import build_coupling


def _two_block_W():
    """Two clearly separated 3-node blocks (strong within, no between)."""
    W = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        W[a, b] = W[b, a] = 1.0
    return W


class TestResolutionGrid:

    def test_inclusive_bounds_and_step(self):
        assert resolution_grid([0.4, 1.2], 0.2) == [0.4, 0.6, 0.8, 1.0, 1.2]

    def test_single_value_when_min_equals_max(self):
        assert resolution_grid([0.7, 0.7], 0.1) == [0.7]

    def test_degenerate_step_returns_min(self):
        assert resolution_grid([0.4, 1.2], 0) == [0.4]


class TestLeidenBestResolution:

    def test_finds_two_communities(self):
        W = _two_block_W()
        labels, Q, res, sweep = leiden_best_resolution(
            W, resolution_grid([0.3, 1.5], 0.3), min_size=2)
        n_comm = len({c for c in labels.tolist() if c >= 0})
        assert n_comm == 2
        assert Q > 0
        assert res in resolution_grid([0.3, 1.5], 0.3)

    def test_sweep_records_every_resolution(self):
        W = _two_block_W()
        grid = resolution_grid([0.3, 1.5], 0.3)
        _, _, _, sweep = leiden_best_resolution(W, grid, min_size=2)
        assert [s[0] for s in sweep] == grid
        assert all(len(triple) == 3 for triple in sweep)  # (res, Q, n_comm)

    def test_prefers_multi_community_over_trivial(self):
        # A single fully-connected block: high resolution splits it (>=2), low
        # keeps it as one. The selector must return a >=2-community partition.
        W = np.ones((6, 6)) - np.eye(6)
        labels, Q, res, sweep = leiden_best_resolution(
            W, resolution_grid([0.2, 3.0], 0.2), min_size=2)
        has_multi = any(n >= 2 for _, _, n in sweep)
        if has_multi:
            assert len({c for c in labels.tolist() if c >= 0}) >= 2


class TestBuildCouplingSweep:

    def _df(self):
        ids = [f"W{i}" for i in range(8)]
        refs = ["a; b; c", "a; b", "a; c", "b; c",
                "x; y; z", "x; y", "x; z", "y; z"]
        return pd.DataFrame({"id": ids, "references": refs})

    def test_sweep_path_records_resolution_and_sweep(self):
        r = build_coupling(self._df(), ref_col="references",
                           resolution_range=[0.3, 1.5], resolution_step=0.3,
                           min_size=2)
        assert r.resolution in resolution_grid([0.3, 1.5], 0.3)
        assert r.resolution_sweep is not None
        assert len(r.resolution_sweep) == len(resolution_grid([0.3, 1.5], 0.3))

    def test_fixed_path_leaves_sweep_none(self):
        r = build_coupling(self._df(), ref_col="references",
                           resolution=0.7, min_size=2)
        assert r.resolution == 0.7
        assert r.resolution_sweep is None


class TestClusterSubthemes:
    """build_coupling populates NetworkResult.terms with per-community TF-IDF."""

    def _df(self):
        rows = []
        blocks = {0: ("solar photovoltaic panel", "rooftop feed-in tariff"),
                  1: ("wind turbine offshore", "capacity factor grid"),
                  2: ("hydrogen electrolysis storage", "fuel cell power")}
        for b, (a, c) in blocks.items():
            pool = [f"{b}_r{k}" for k in range(4)]
            for i in range(10):
                rows.append({"id": f"W{b}_{i}",
                             "title": a + " energy transition",
                             "abstract": c + " policy model",
                             "references": "; ".join(pool[:3])})
        return pd.DataFrame(rows)

    def test_terms_populated_per_community(self):
        r = build_coupling(self._df(), ref_col="references",
                           resolution_range=[0.3, 1.5], resolution_step=0.2,
                           min_size=3)
        assert r.terms                                   # non-empty
        assert set(r.terms) <= set(int(c) for c in np.unique(r.labels) if c >= 0)
        # Distinct blocks yield distinct top terms.
        joined = {c: " ".join(t) for c, t in r.terms.items()}
        assert any("solar" in v for v in joined.values())
        assert any("wind" in v or "turbine" in v for v in joined.values())

    def test_missing_text_columns_give_empty_terms(self):
        df = pd.DataFrame({"id": [f"W{i}" for i in range(6)],
                           "references": ["a; b", "a; b", "a; c",
                                          "x; y", "x; y", "x; z"]})
        r = build_coupling(df, ref_col="references", resolution=0.7, min_size=2)
        assert r.terms == {}
