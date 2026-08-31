"""Tests for coupling-community LLM labels: the fingerprint cache and the
async labeller. The cache must be keyed on the sub-theme terms themselves so the
LLM runs once and never again on an unchanged Leiden partition.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.topic_labels import (
    terms_fingerprint,
    load_cached_cluster_labels,
    save_cluster_labels,
)


class TestTermsFingerprint:

    def test_stable_and_order_independent(self):
        a = {0: ["solar", "pv"], 1: ["wind", "turbine"]}
        b = {1: ["wind", "turbine"], 0: ["solar", "pv"]}   # keys reordered
        assert terms_fingerprint(a) == terms_fingerprint(b)

    def test_changes_when_terms_change(self):
        a = {0: ["solar", "pv"]}
        b = {0: ["solar", "photovoltaic"]}
        assert terms_fingerprint(a) != terms_fingerprint(b)

    def test_int_and_str_keys_agree(self):
        assert terms_fingerprint({0: ["x"]}) == terms_fingerprint({"0": ["x"]})


class TestClusterLabelsCache:

    def test_miss_then_hit_roundtrip(self, tmp_path):
        run_dir = str(tmp_path)
        terms = {0: ["solar", "pv"], 1: ["wind"]}
        assert load_cached_cluster_labels(run_dir, terms) is None   # miss
        save_cluster_labels(run_dir, terms, {0: "Solar", 1: "Wind"})
        cached = load_cached_cluster_labels(run_dir, terms)
        assert cached == {0: "Solar", 1: "Wind"}                    # int keys back

    def test_different_terms_do_not_collide(self, tmp_path):
        run_dir = str(tmp_path)
        save_cluster_labels(run_dir, {0: ["solar"]}, {0: "Solar"})
        # A different partition (different terms) is a cache miss, not a stale hit.
        assert load_cached_cluster_labels(run_dir, {0: ["wind"]}) is None


class _FakeProvider:
    """Returns a label derived from the message content so we can assert the
    terms actually reached the prompt."""

    async def call(self, messages, model_id, model_args, response_schema):
        user = messages[-1]["content"]
        return {"label": f"label::{user[:20]}"}, None


class TestLabelClusters:

    def _config(self):
        from pysyrev.core.config import TopicLabelerConfig
        return TopicLabelerConfig(provider="fake", model_id="m")

    def test_labels_every_community(self, monkeypatch):
        from pysyrev.core import llm
        monkeypatch.setattr(llm, "_make_provider", lambda *a, **k: _FakeProvider())
        terms = {0: ["solar", "pv"], 2: ["wind", "turbine"]}
        out = llm.label_clusters(terms, self._config())
        assert set(out) == {0, 2}
        assert all(v.startswith("label::") for v in out.values())

    def test_repr_titles_reach_the_prompt(self, monkeypatch):
        from pysyrev.core import llm
        captured = {}

        class _Capture:
            async def call(self, messages, model_id, model_args, schema):
                captured["user"] = messages[-1]["content"]
                return {"label": "ok"}, None

        monkeypatch.setattr(llm, "_make_provider", lambda *a, **k: _Capture())
        llm.label_clusters({0: ["solar"]}, self._config(),
                           repr_docs={0: ["A study of rooftop solar"]})
        assert "rooftop solar" in captured["user"]
        assert "solar" in captured["user"]      # terms present too


class TestCommunityReprTitles:

    def test_top_strength_titles_per_community(self):
        from pysyrev.core.report_data import _community_repr_titles
        from pysyrev.core.networks.common import NetworkResult
        ids = ["W0", "W1", "W2", "W3"]
        # W0 strongly connected within comm 0; comm 1 = {W2, W3}.
        W = np.array([[0, 2, 0, 0], [2, 0, 0, 0],
                      [0, 0, 0, 1], [0, 0, 1, 0]], float)
        res = NetworkResult(node_ids=ids, W=W, labels=np.array([0, 0, 1, 1]),
                            coords=np.zeros((4, 2)), modularity=0.3)
        df = pd.DataFrame({"id": ids,
                           "title": ["Solar A", "Solar B", "Wind A", "Wind B"]})
        out = _community_repr_titles(res, df, n=5)
        assert out[0][0] in ("Solar A", "Solar B")
        assert set(out[1]) == {"Wind A", "Wind B"}
