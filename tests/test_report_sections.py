"""
Tests for the pysyrev.core.report section builders.

All tests are fast: they use synthetic DataFrames from conftest and do not
require any ML model or disk access beyond tmp_path.

Test naming convention:
  test_<what>_<condition>
"""

import os

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.config import (
    BibNetworkSectionConfig,
    CocitationPanelConfig,
    CompositeConfig,
    CouplingPanelConfig,
    PaperSelectionConfig,
    TemporalSectionConfig,
    TopicCharacteristicsConfig,
    TopicsSectionConfig,
    TopicSimilarityConfig,
)
from pysyrev.core.report import build_report_data
from pysyrev.core.report.data import _build_overview_section
from pysyrev.core.report.sections.networks import _build_networks_section
from pysyrev.core.report.sections.papers import _build_paper_selection_section
from pysyrev.core.report.sections.temporal import _build_temporal_section
from pysyrev.core.report.sections.topics import (
    _build_topic_characteristics_section,
    _build_topic_similarity_section,
    _build_topics_section,
)


# =============================================================================
# Helpers
# =============================================================================

def _blocks_of_type(section, block_type):
    """Collect all blocks of a given type, including inside subsections."""
    results = []
    for block in section.get("blocks", []):
        if block.get("type") == block_type:
            results.append(block)
        for sub in block.get("blocks", []):
            if sub.get("type") == block_type:
                results.append(sub)
    return results


def _subsections(section):
    return [b for b in section.get("blocks", []) if b.get("type") == "subsection"]


# =============================================================================
# _build_topics_section
# =============================================================================

class TestBuildTopicsSection:

    def _run(self, topic_info, sections_cfg=None, topic_labels=None, nb_topics=3,
             n=1, bertopic_results=None):
        cfg = sections_cfg or TopicsSectionConfig(n_repr_docs_per_topic=3)

        class _FakeSections:
            topics = cfg

        return _build_topics_section(topic_info, _FakeSections(), topic_labels,
                                     nb_topics, n, bertopic_results=bertopic_results)

    def test_returns_section_dict(self, tiny_topic_info):
        sec = self._run(tiny_topic_info)
        assert isinstance(sec, dict)
        assert "title" in sec and "blocks" in sec

    def test_title_contains_section_number_and_topics(self, tiny_topic_info):
        sec = self._run(tiny_topic_info, n=1)
        assert "1." in sec["title"]
        assert "Topic" in sec["title"]

    def test_summary_table_is_first_block(self, tiny_topic_info):
        sec = self._run(tiny_topic_info)
        assert sec["blocks"][0]["type"] == "table"

    def test_summary_table_has_one_row_per_topic(self, tiny_topic_info, nb_topics=3):
        sec = self._run(tiny_topic_info, nb_topics=nb_topics)
        table = sec["blocks"][0]
        assert len(table["rows"]) == nb_topics

    def test_one_subsection_per_topic(self, tiny_topic_info):
        sec = self._run(tiny_topic_info)
        assert len(_subsections(sec)) == 3

    def test_subsection_title_uses_default_label_without_llm(self, tiny_topic_info):
        sec = self._run(tiny_topic_info, topic_labels=None)
        sub_titles = [s["title"] for s in _subsections(sec)]
        assert any("0" in t for t in sub_titles)

    def test_subsection_title_uses_llm_label(self, tiny_topic_info):
        labels = {0: "Agent-Based Models", 1: "Epidemiology", 2: "Network Theory"}
        sec = self._run(tiny_topic_info, topic_labels=labels)
        sub_titles = [s["title"] for s in _subsections(sec)]
        assert any("Agent-Based Models" in t for t in sub_titles)

    def test_n_repr_docs_per_topic_limits_table_rows(self, tiny_topic_info):
        # n_repr_docs_per_topic=2 → each per-topic doc table has at most 2 rows
        sec = self._run(tiny_topic_info, sections_cfg=TopicsSectionConfig(n_repr_docs_per_topic=2))
        for sub in _subsections(sec):
            doc_tables = _blocks_of_type(sub, "table")
            for t in doc_tables:
                assert len(t["rows"]) <= 2

    def test_missing_repr_doc_columns_does_not_crash(self, tiny_topic_info):
        # Drop all repr_doc_* columns
        stripped = tiny_topic_info.drop(
            columns=[c for c in tiny_topic_info.columns if c.startswith("repr_doc_")]
        )
        sec = self._run(stripped)
        assert len(_subsections(sec)) == 3

    def test_summary_table_uses_llm_label_column_header(self, tiny_topic_info):
        labels = {0: "L0", 1: "L1", 2: "L2"}
        sec = self._run(tiny_topic_info, topic_labels=labels)
        table = sec["blocks"][0]
        assert "LLM label" in table["headers"]

    def test_summary_table_uses_name_column_header_without_llm(self, tiny_topic_info):
        sec = self._run(tiny_topic_info, topic_labels=None)
        table = sec["blocks"][0]
        assert "Name" in table["headers"]

    def test_no_median_column_without_bertopic_results(self, tiny_topic_info):
        table = self._run(tiny_topic_info)["blocks"][0]
        assert "Docs/yr" not in table["headers"]
        assert all(len(row) == len(table["headers"]) for row in table["rows"])

    def test_median_column_added_with_bertopic_results(self, tiny_topic_info,
                                                       tiny_bertopic_results):
        table = self._run(tiny_topic_info,
                          bertopic_results=tiny_bertopic_results)["blocks"][0]
        assert table["headers"] == ["ID", "Name", "# Docs", "Docs/yr",
                                    "Med. year", "Top words"]
        assert len(table["col_widths"]) == len(table["headers"])
        assert all(len(row) == len(table["headers"]) for row in table["rows"])
        assert table["caption"]

    def test_median_columns_dropped_without_year(self, tiny_topic_info,
                                                 tiny_bertopic_results):
        br = tiny_bertopic_results.drop(columns=["year"])
        table = self._run(tiny_topic_info, bertopic_results=br)["blocks"][0]
        assert "Docs/yr" not in table["headers"]
        assert "Med. year" not in table["headers"]

    def test_median_is_the_median_of_the_topic_own_years(self, tiny_topic_info):
        # Topic 0: 3 docs in 2020, 1 in 2021, none in 2022 → median over
        # 2020-2022 is 1. Topic 1 starts in 2022, so its two silent years
        # before that must not count: median over 2022 alone is 4.
        br = pd.DataFrame({
            "Topic": [0] * 4 + [1] * 4 + [-1],
            "year":  [2020, 2020, 2020, 2021] + [2022] * 4 + [2020],
        })
        table = self._run(tiny_topic_info, bertopic_results=br)["blocks"][0]
        by_id = {row[0]: row[3] for row in table["rows"]}
        assert by_id["0"] == "1"
        assert by_id["1"] == "4"
        assert by_id["2"] == "-"        # a topic with no document at all

    def test_median_year_is_where_the_topic_sits_in_time(self, tiny_topic_info):
        # Topic 0 publishes 4 documents in 2020 and one in 2024: it is a 2020
        # topic, not a 2022 one. Topic 1's four documents are all from 2024.
        br = pd.DataFrame({
            "Topic": [0] * 5 + [1] * 4 + [-1],
            "year":  [2020, 2020, 2020, 2020, 2024] + [2024] * 4 + [2020],
        })
        table = self._run(tiny_topic_info, bertopic_results=br)["blocks"][0]
        by_id = {row[0]: row[4] for row in table["rows"]}
        assert by_id["0"] == "2020"
        assert by_id["1"] == "2024"
        assert by_id["2"] == "-"        # a topic with no document at all

    def test_median_year_is_taken_low_on_an_even_count(self, tiny_topic_info):
        # Two documents, 2010 and 2013 → a real year of the corpus, not 2011.5
        br = pd.DataFrame({"Topic": [0, 0], "year": [2010, 2013]})
        table = self._run(tiny_topic_info, bertopic_results=br)["blocks"][0]
        assert table["rows"][0][4] == "2010"


# =============================================================================
# _build_networks_section  (coupling + co-citation, recomputed from references)
# =============================================================================

class TestBuildNetworksSection:

    @staticmethod
    def _results(path):
        import pandas as pd
        from pysyrev.core.networks import build_coupling, build_cocitation
        df = pd.read_csv(path, low_memory=False)
        coupling   = build_coupling(df, resolution=0.7, min_size=3)
        cocitation = build_cocitation(df, min_ref_freq=2, resolution=0.7, min_size=3)
        return df, coupling, cocitation

    def test_returns_none_without_results(self):
        cfg = BibNetworkSectionConfig()
        assert _build_networks_section(None, None, None, None, None, cfg, None, 2) is None

    def test_both_network_subsections_present(self, reviewed_dataset_path,
                                              tiny_bertopic_results):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3), cocitation=CocitationPanelConfig(min_size=3))
        df, coupling, cocitation = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, cocitation,
                                      tiny_bertopic_results, None, cfg, None, 2)
        assert sec is not None
        titles = [s["title"] for s in _subsections(sec)]
        assert "Bibliographic coupling" in titles
        assert "Co-citation" in titles

    def test_title_contains_section_number(self, reviewed_dataset_path):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3), cocitation=CocitationPanelConfig(min_size=3))
        df, coupling, cocitation = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, cocitation, None, None, cfg, None, 3)
        assert "3." in sec["title"]

    def test_coupling_subsection_has_stats_and_table(self, reviewed_dataset_path):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3))
        df, coupling, _ = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, None, None, None, cfg, None, 2)
        subs = _subsections(sec)
        assert subs and subs[0]["title"] == "Bibliographic coupling"
        kv = _blocks_of_type(subs[0], "key_value")
        assert kv
        keys = {item["key"] for item in kv[0]["items"]}
        assert {"Documents", "Modularity"}.issubset(keys)
        assert _blocks_of_type(subs[0], "table")

    def test_no_topic_connectivity_without_topics(self, reviewed_dataset_path):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3))
        df, coupling, _ = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, None, None, None, cfg, None, 2)
        titles = [s["title"] for s in _subsections(sec)]
        assert "Bibliographic coupling" in titles
        assert "Topic connectivity" not in titles      # no topics passed
        # Community connectivity is topic-independent and may still appear.

    def test_both_connectivity_panels_with_matching_topics(self, reviewed_dataset_path):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3))
        df, coupling, _ = self._results(reviewed_dataset_path)
        # BERTopic-style results whose ids match the corpus → topic groups exist
        ids = list(df["id"])
        bt = pd.DataFrame({"id": ids, "Topic": [i % 3 for i in range(len(ids))]})
        sec = _build_networks_section(df, coupling, None, bt, None, cfg, None, 2)
        titles = [s["title"] for s in _subsections(sec)]
        assert "Topic connectivity" in titles
        assert "Community connectivity (coupling)" in titles

    def test_community_connectivity_is_rendered_per_network(self,
                                                            reviewed_dataset_path):
        # coupling and co-citation communities answer different questions, so
        # each gets its own panel
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3),
                                      cocitation=CocitationPanelConfig(min_size=3))
        df, coupling, cocitation = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, cocitation, None, None, cfg,
                                      None, 2)
        titles = [s["title"] for s in _subsections(sec)]
        assert "Community connectivity (coupling)" in titles
        assert "Community connectivity (co-citation)" in titles

    def test_no_cocitation_connectivity_without_a_cocitation_network(
            self, reviewed_dataset_path):
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3))
        df, coupling, _ = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, None, None, None, cfg, None, 2)
        titles = [s["title"] for s in _subsections(sec)]
        assert "Community connectivity (coupling)" in titles
        assert "Community connectivity (co-citation)" not in titles

    def test_cocitation_connectivity_uses_its_own_vocabulary(self,
                                                             reviewed_dataset_path):
        # the matrix machinery is shared, the prose is not: a co-citation panel
        # must not describe itself as bibliographic coupling
        cfg = BibNetworkSectionConfig(coupling=CouplingPanelConfig(min_size=3),
                                      cocitation=CocitationPanelConfig(min_size=3))
        df, coupling, cocitation = self._results(reviewed_dataset_path)
        sec = _build_networks_section(df, coupling, cocitation, None, None, cfg,
                                      None, 2)
        sub = next(s for s in _subsections(sec)
                   if s["title"] == "Community connectivity (co-citation)")
        text = _blocks_of_type(sub, "paragraph")[0]["text"]
        assert "co-citation" in text
        assert "bibliographic coupling" not in text
        assert _blocks_of_type(sub, "table")[0]["title"] == (
            "Internal vs outward co-citation per community (obs/exp)")


class TestConnectivitySupportThreshold:

    @staticmethod
    def _panel(min_support, observations):
        from pysyrev.core.report.sections.networks import _connectivity_panel
        W = np.array([[0, .9, .1, 0],
                      [.9, 0, 0, .1],
                      [.1, 0, 0, .9],
                      [0, .1, .9, 0]], dtype=float)
        groups = [("A", np.array([0, 1])), ("B", np.array([2, 3]))]
        return _connectivity_panel(W, groups, "community", None,
                                   kind="cocitation", observations=observations,
                                   min_support=min_support)

    @staticmethod
    def _table(panel):
        return next(b for b in panel["blocks"] if b["type"] == "table")

    def test_a_thin_group_reports_no_internal_ratio(self):
        # two observations back A, six back B: A's internal ratio is unknown,
        # not small, so it must not be printed as a number that can be ranked
        obs = [[0, 1], [0, 1]] + [[2, 3]] * 6
        table = self._table(self._panel(5, obs))
        rows = {r[0]: r for r in table["rows"]}
        assert rows["A"][2] == "—"
        assert rows["A"][-1] == "insufficient"
        assert rows["B"][2] != "—"
        assert rows["B"][-1] in {"self-contained", "integrated"}

    def test_the_withheld_ratio_is_a_gap_in_the_heatmap_diagonal(self):
        obs = [[0, 1], [0, 1]] + [[2, 3]] * 6
        panel = self._panel(5, obs)
        z = np.asarray(next(b for b in panel["blocks"]
                            if b["type"] == "plotly")["figure"].data[0].z,
                       dtype=float)
        assert not np.isfinite(z[0, 0])       # A withheld
        assert np.isfinite(z[1, 1])           # B reported
        assert np.isfinite(z[0, 1])           # outward cells are never withheld

    def test_outward_still_reported_for_a_thin_group(self):
        # only the diagonal rests on the group's own thin evidence; its outward
        # cells rest on the whole corpus and stay readable
        obs = [[0, 1], [0, 1]] + [[2, 3]] * 6
        rows = {r[0]: r for r in self._table(self._panel(5, obs))["rows"]}
        assert rows["A"][3] != "—"

    def test_threshold_is_honoured(self):
        obs = [[0, 1], [0, 1]] + [[2, 3]] * 6
        lenient = {r[0]: r for r in self._table(self._panel(2, obs))["rows"]}
        strict  = {r[0]: r for r in self._table(self._panel(20, obs))["rows"]}
        assert lenient["A"][2] != "—" and lenient["B"][2] != "—"
        assert strict["A"][2] == "—" and strict["B"][2] == "—"

    def test_nothing_is_withheld_without_observations(self):
        table = self._table(self._panel(1000, None))
        assert all(r[2] != "—" for r in table["rows"])
        assert "Support" not in table["headers"]


class TestConnectivityMatrix:

    def test_matrix_symmetric_scaled_and_baseline(self):
        from pysyrev.core.networks import (
            inter_cluster_matrix, corpus_baseline, coupling_inout,
        )
        W = np.array([[0, .4, .1, 0],
                      [.4, 0, 0, .2],
                      [.1, 0, 0, .3],
                      [0, .2, .3, 0]], dtype=float)
        groups = [("A", np.array([0, 1])), ("B", np.array([2, 3]))]
        M, labs = inter_cluster_matrix(W, groups, scale=1000)
        assert labs == ["A", "B"]
        assert np.allclose(M, M.T)
        assert M[0, 0] == 400.0          # single intra-A pair W=0.4 → ×1000
        assert corpus_baseline(W, scale=1000) > 0
        prof = coupling_inout(M, 0)
        assert prof["internal"] == M[0, 0]
        assert prof["outward"] == M[0, 1]

    def test_a_block_at_the_null_scores_one_whatever_the_group_size(self):
        # a group carrying exactly the weight its nodes' strengths predict
        # scores 1.00, whether it holds 5 strong nodes or 100 weak ones. The
        # raw mean ranks the two apart though neither is enriched at all.
        from pysyrev.core.networks import (configuration_null,
                                           inter_cluster_matrix,
                                           inter_cluster_enrichment)
        strengths = np.concatenate([np.full(5, 40.0), np.full(100, 2.0)])
        W = np.outer(strengths, strengths) / strengths.sum()
        np.fill_diagonal(W, 0.0)
        W = configuration_null(W)        # a network that *is* its own null
        groups = [("strong", np.arange(5)), ("weak", np.arange(5, 105))]

        M, _ = inter_cluster_matrix(W, groups, scale=1000)
        R, labs = inter_cluster_enrichment(W, groups)
        assert labs == ["strong", "weak"]
        assert np.allclose(R, R.T)
        assert M[0, 0] / M[1, 1] > 50            # the mean ranks them apart...
        assert R[0, 0] == pytest.approx(1.0, rel=1e-6)   # ...the ratio does not
        assert R[1, 1] == pytest.approx(1.0, rel=1e-6)

    def test_enrichment_separates_real_cohesion_from_a_saturated_micro_group(self):
        # the case that prompted the change, in miniature: a few references
        # co-cited by two documents against two much larger intellectual bases.
        # The ratio must score the equal-sized bases equally; the support count
        # is what exposes the micro-group.
        from pysyrev.core.networks import (cocitation_matrix, internal_support,
                                           inter_cluster_matrix,
                                           inter_cluster_enrichment)
        rng = np.random.default_rng(0)
        micro = [f"a{i}" for i in range(8)]
        base_b = [f"b{i}" for i in range(300)]
        base_c = [f"c{i}" for i in range(300)]
        refsets = [set(micro), set(micro)]           # two documents, that is all
        for src, other in ((base_b, base_c), (base_c, base_b)):
            for _ in range(400):
                refsets.append(set(rng.choice(src, 20, replace=False)) |
                               set(rng.choice(other, 2, replace=False)))

        ids, W = cocitation_matrix(refsets, min_ref_freq=2)
        pos = {r: i for i, r in enumerate(ids)}
        groups = [(name, np.array([pos[r] for r in members if r in pos]))
                  for name, members in (("micro", micro), ("B", base_b),
                                        ("C", base_c))]
        M, _ = inter_cluster_matrix(W, groups, scale=1000)
        R, _ = inter_cluster_enrichment(W, groups)
        observations = [[pos[r] for r in refs if r in pos] for refs in refsets]
        support = internal_support([o for o in observations if len(o) > 1], groups)

        # the two bases are built the same way, so they must score the same
        assert R[1, 1] == pytest.approx(R[2, 2], rel=0.15)
        # each is cohesive on its own terms and depleted towards the other
        assert R[1, 1] > 1 and R[1, 2] < 1
        assert M[0, 0] / M[1, 1] > 10        # the raw mean buried them
        # the micro-group still scores high, so support is what exposes it
        assert support[0] == 2
        assert support[1] > 100 and support[2] > 100

    def test_enrichment_matches_averaging_the_dense_null(self):
        # inter_cluster_enrichment evaluates the null's block means in closed
        # form to avoid building a second dense n x n matrix; it must agree
        # exactly with averaging configuration_null itself.
        from pysyrev.core.networks import (configuration_null,
                                           inter_cluster_matrix,
                                           inter_cluster_enrichment)
        rng = np.random.default_rng(3)
        for _ in range(3):
            n = 40
            A = np.triu(rng.random((n, n)) * (rng.random((n, n)) < 0.4), 1)
            W = A + A.T
            perm = rng.permutation(n)
            groups = [("a", perm[:9]), ("b", perm[9:20]), ("c", perm[20:])]
            obs, _ = inter_cluster_matrix(W, groups, scale=1.0)
            exp, _ = inter_cluster_matrix(configuration_null(W), groups, scale=1.0)
            expected = np.where(exp > 0, obs / exp, 0.0)
            got, _ = inter_cluster_enrichment(W, groups)
            assert np.allclose(got, expected)

    def test_configuration_null_preserves_total_strength(self):
        from pysyrev.core.networks import configuration_null
        W = np.array([[0, 2., 1.], [2., 0, 3.], [1., 3., 0]])
        E = configuration_null(W)
        assert np.allclose(E.sum(axis=1), W.sum(axis=1))
        assert np.allclose(configuration_null(np.zeros((3, 3))), 0.0)

    def test_internal_support_counts_observations_touching_two_nodes(self):
        from pysyrev.core.networks import internal_support
        groups = [("A", np.array([0, 1, 2])), ("B", np.array([3, 4]))]
        observations = [
            [0, 1],        # supports A
            [0, 2, 3],     # supports A (two of its nodes), not B (only one)
            [3, 4],        # supports B
            [2],           # supports nothing: a single node makes no pair
        ]
        assert internal_support(observations, groups) == [2, 1]


class TestCompositeSelection:

    @staticmethod
    def _setup(path):
        from pysyrev.core.networks import build_coupling
        df = pd.read_csv(path, low_memory=False)
        coupling = build_coupling(df, resolution=0.7, min_size=3)
        ids = list(df["id"])
        bt = pd.DataFrame({
            "id": ids,
            "Document": (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist(),
            "Topic": [i % 3 for i in range(len(ids))],
            "year": df["year"].tolist(),
            "cited_by": df["cited_by"].tolist(),
            "document_type": df["document_type"].tolist(),
            "title": df["title"].tolist(),
            "doi": df["doi"].tolist(),
        })
        return coupling, bt

    def test_scores_carry_three_axes(self, reviewed_dataset_path):
        from pysyrev.core.report.sections.papers import _composite_scores
        coupling, bt = self._setup(reviewed_dataset_path)
        detail = _composite_scores(coupling, bt, aggregate="mean")
        assert detail
        assert all(0.0 <= d["score"] <= 1.0 for d in detail.values())
        d0 = next(iter(detail.values()))
        assert {"centrality", "relevance", "representativeness"} <= set(d0)

    def test_empty_without_coupling(self):
        from pysyrev.core.report.sections.papers import _composite_scores
        assert _composite_scores(None, None) == {}

    def test_composite_table_shows_axis_columns(self, reviewed_dataset_path):
        from pysyrev.core.report.sections.papers import _build_paper_selection_section
        coupling, bt = self._setup(reviewed_dataset_path)
        cfg = PaperSelectionConfig(min_year=2000, proportion_per_topic=0.5,
                                   selection_by="composite", export_annex=False)
        sec = _build_paper_selection_section(bt, None, None, cfg, None, 6,
                                             coupling_result=coupling)
        assert sec is not None
        table = _blocks_of_type(sec, "table")[0]
        assert {"Score", "Cent.", "Rel.", "Typ."} <= set(table["headers"])
        assert any(r[-1] == "Most relevant (3-axis)" for r in table["rows"])

    def test_tuning_params_change_scores(self, reviewed_dataset_path):
        from pysyrev.core.report.sections.papers import _composite_scores
        coupling, bt = self._setup(reviewed_dataset_path)
        base = _composite_scores(coupling, bt, aggregate="mean")
        cpy  = _composite_scores(coupling, bt, aggregate="mean", relevance_mode="cpy")
        wrel = _composite_scores(coupling, bt, aggregate="mean", weights=[1, 1, 5])
        assert any(base[k]["score"] != cpy[k]["score"] for k in base)
        assert any(base[k]["score"] != wrel[k]["score"] for k in base)

    def test_weights_validation(self):
        with pytest.raises(ValueError):
            CompositeConfig(weights=[1, 1])

    def test_split_current_year_adds_a_research_fronts_table(self):
        # Synthetic corpus with a clear current-year cohort so the split is
        # deterministic regardless of the fixture data's years.
        from pysyrev.core.report.sections.papers import _build_paper_selection_section
        from pysyrev.core.networks.common import NetworkResult
        ids = [f"W{i}" for i in range(6)]
        bt = pd.DataFrame({
            "id": ids, "ID": ids,
            "year": [2019, 2020, 2021, 2022, 2026, 2026],
            "cited_by": [80, 60, 40, 20, 3, 1],
            "document_type": ["article"] * 6,
            "title": [f"Paper {i}" for i in range(6)],
            "doi": [f"10.1/{i}" for i in range(6)],
            "Topic": [0] * 6,
            "Document": [f"energy justice model {i}" for i in range(6)],
        })
        W = np.ones((6, 6)) - np.eye(6)
        coupling = NetworkResult(node_ids=ids, W=W, labels=np.zeros(6, int),
                                 coords=np.zeros((6, 2)), modularity=0.1)
        cfg = PaperSelectionConfig(
            min_year=2015, proportion_per_topic=1.0, selection_by="composite",
            export_annex=False,
            composite=CompositeConfig(split_current_year=True, current_year=2026))
        sec = _build_paper_selection_section(bt, None, None, cfg, None, 6,
                                             coupling_result=coupling)
        tables = _blocks_of_type(sec, "table")
        titles = [t["title"] for t in tables]
        assert any(t.startswith("Paper reading list") for t in titles)
        fronts = next(t for t in tables if t["title"].startswith("Research fronts"))
        # Fronts are 2-axis: no relevance column, and only current-year rows.
        assert "Rel." not in fronts["headers"]
        assert all(r[3] == "2026" for r in fronts["rows"])   # Year column
        # Historical list excludes the current-year fronts.
        reading = next(t for t in tables if t["title"].startswith("Paper reading list"))
        assert all(r[3] != "2026" for r in reading["rows"])  # Year column

    def test_csv_annex_holds_both_cohorts_with_a_cohort_column(self, tmp_path):
        from pysyrev.core.report.sections.papers import _build_paper_selection_section
        from pysyrev.core.networks.common import NetworkResult
        ids = [f"W{i}" for i in range(6)]
        bt = pd.DataFrame({
            "id": ids, "ID": ids,
            "year": [2019, 2020, 2021, 2022, 2026, 2026],
            "cited_by": [80, 60, 40, 20, 3, 1],
            "document_type": ["article"] * 6,
            "title": [f"Paper {i}" for i in range(6)],
            "doi": [f"10.1/{i}" for i in range(6)],
            "Topic": [0] * 6,
            "Document": [f"energy model {i}" for i in range(6)],
        })
        W = np.ones((6, 6)) - np.eye(6)
        coupling = NetworkResult(node_ids=ids, W=W, labels=np.zeros(6, int),
                                 coords=np.zeros((6, 2)), modularity=0.1)
        cfg = PaperSelectionConfig(
            min_year=2015, proportion_per_topic=1.0, selection_by="composite",
            export_annex=True, annex_format="csv",
            composite=CompositeConfig(split_current_year=True, current_year=2026))
        _build_paper_selection_section(bt, None, None, cfg, str(tmp_path), 6,
                                       coupling_result=coupling)
        csv = pd.read_csv(tmp_path / "paper_selection.csv")
        assert "cohort" in csv.columns
        assert set(csv["cohort"]) == {"historical", "front"}
        # The two 2026 papers are the fronts; the rest are historical.
        assert set(csv[csv["cohort"] == "front"]["year"]) == {2026}
        assert 2026 not in set(csv[csv["cohort"] == "historical"]["year"])

    def test_fronts_are_a_proportional_selection_not_all(self, tmp_path):
        # Many current-year papers, small proportion → the CSV fronts are a
        # top-N-per-topic selection, not the whole current year, and match the
        # research-fronts table count.
        from pysyrev.core.report.sections.papers import _build_paper_selection_section
        from pysyrev.core.networks.common import NetworkResult
        n = 30
        ids = [f"W{i}" for i in range(n)]
        years = [2020 + (i % 5) for i in range(20)] + [2026] * 10   # 20 hist, 10 front
        bt = pd.DataFrame({
            "id": ids, "ID": ids, "year": years,
            "cited_by": [50 - i for i in range(n)],
            "document_type": ["article"] * n,
            "title": [f"Paper {i}" for i in range(n)],
            "doi": [f"10.1/{i}" for i in range(n)],
            "Topic": [0] * n,
            "Document": [f"energy model {i}" for i in range(n)],
        })
        W = np.ones((n, n)) - np.eye(n)
        coupling = NetworkResult(node_ids=ids, W=W, labels=np.zeros(n, int),
                                 coords=np.zeros((n, 2)), modularity=0.1)
        cfg = PaperSelectionConfig(
            min_year=2015, proportion_per_topic=0.2, selection_by="composite",
            export_annex=True, annex_format="csv",
            composite=CompositeConfig(split_current_year=True, current_year=2026))
        sec = _build_paper_selection_section(bt, None, None, cfg, str(tmp_path), 6,
                                             coupling_result=coupling)
        csv = pd.read_csv(tmp_path / "paper_selection.csv")
        n_front_csv = int((csv["cohort"] == "front").sum())
        assert 0 < n_front_csv < 10                      # a selection, not all 10 fronts
        fronts_table = next(t for t in _blocks_of_type(sec, "table")
                            if t["title"].startswith("Research fronts"))
        assert len(fronts_table["rows"]) == n_front_csv  # table and CSV agree


# =============================================================================
# _build_temporal_section
# =============================================================================

class TestBuildTemporalSection:

    def test_returns_none_without_year_column(self, tiny_bertopic_results):
        br = tiny_bertopic_results.drop(columns=["year"])
        cfg = TemporalSectionConfig(variants=["absolute"])
        assert _build_temporal_section(br, 3, cfg, 3) is None

    def test_returns_section_dict(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["absolute"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        assert isinstance(sec, dict) and "blocks" in sec

    def test_one_subsection_per_variant(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["absolute", "normalized"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        assert len(_subsections(sec)) == 2

    def test_all_variants_produce_a_subsection_each(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(
            variants=["absolute", "normalized", "weighted"]
        )
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        assert len(_subsections(sec)) == 3

    def test_each_subsection_has_plotly_block(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["absolute", "normalized"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        for sub in _subsections(sec):
            assert _blocks_of_type(sub, "plotly"), f"No plotly block in sub: {sub['title']}"

    def test_weighted_variant_skipped_without_topic_cols(self, tiny_bertopic_results):
        br = tiny_bertopic_results.drop(
            columns=[c for c in tiny_bertopic_results.columns if c.startswith("topic#")]
        )
        cfg = TemporalSectionConfig(variants=["absolute", "weighted"])
        sec = _build_temporal_section(br, 3, cfg, 3)
        # "weighted" skipped silently → only "absolute" subsection remains
        assert len(_subsections(sec)) == 1

    def test_outliers_do_not_appear_in_absolute_figure_data(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["absolute"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        sub = _subsections(sec)[0]
        fig = _blocks_of_type(sub, "plotly")[0]["figure"]
        # The figure's data traces must not contain Topic -1
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name is not None:
                assert str(trace.name) != "-1"

    def test_title_contains_section_number(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["absolute"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 4)
        assert "4." in sec["title"]

    def test_trajectory_is_a_default_variant(self):
        assert "trajectory" in TemporalSectionConfig().variants

    def test_smoothing_window_below_one_is_refused(self):
        with pytest.raises(ValueError):
            TemporalSectionConfig(smoothing_window=0)

    def test_trajectory_draws_counts_smoothed_and_share(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["trajectory"], smoothing_window=3)
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        sub = _subsections(sec)[0]
        assert "per year" in sub["title"].lower()
        assert len(_blocks_of_type(sub, "plotly")) == 3

    def test_trajectory_without_smoothing_drops_the_rolling_mean(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(variants=["trajectory"], smoothing_window=1)
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        sub = _subsections(sec)[0]
        assert len(_blocks_of_type(sub, "plotly")) == 2

    def test_trajectory_curves_are_not_cumulative(self, tiny_bertopic_results):
        # The whole point of the section: a topic that stops publishing must be
        # able to come back down, which a cumulative curve can never do.
        cfg = TemporalSectionConfig(variants=["trajectory"], smoothing_window=1)
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        fig = _blocks_of_type(_subsections(sec)[0], "plotly")[0]["figure"]
        assert any(
            any(b < a for a, b in zip(trace.y, trace.y[1:]))
            for trace in fig.data
        )

    def test_trajectory_uses_llm_labels_in_the_legend(self, tiny_bertopic_results):
        labels = {0: "Agent-Based Models", 1: "Epidemiology", 2: "Network Theory"}
        cfg = TemporalSectionConfig(variants=["trajectory"], smoothing_window=1)
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3,
                                      topic_labels=labels)
        fig = _blocks_of_type(_subsections(sec)[0], "plotly")[0]["figure"]
        assert {t.name for t in fig.data} == {
            "0. Agent-Based Models", "1. Epidemiology", "2. Network Theory"}

    def test_trajectory_skipped_on_a_single_year_corpus(self, tiny_bertopic_results):
        br = tiny_bertopic_results.assign(year=2020)
        cfg = TemporalSectionConfig(variants=["trajectory", "absolute"])
        sec = _build_temporal_section(br, 3, cfg, 3)
        # no trend to draw through one point → only the "absolute" subsection
        assert [s["title"] for s in _subsections(sec)] == ["Absolute"]

    def test_trajectory_fills_the_years_a_topic_skipped(self, tiny_bertopic_results):
        br = tiny_bertopic_results[tiny_bertopic_results["year"] != 2015]
        cfg = TemporalSectionConfig(variants=["trajectory"], smoothing_window=1)
        sec = _build_temporal_section(br, 3, cfg, 3)
        fig = _blocks_of_type(_subsections(sec)[0], "plotly")[0]["figure"]
        for trace in fig.data:
            assert 2015 in list(trace.x)          # drawn at zero, not skipped
            assert len(trace.x) == len(set(trace.x))


# =============================================================================
# _build_topic_characteristics_section
# =============================================================================

class TestBuildTopicCharacteristicsSection:

    def test_returns_section_dict(self, tiny_bertopic_results):
        cfg = TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5)
        sec = _build_topic_characteristics_section(tiny_bertopic_results, None, cfg, 4)
        assert isinstance(sec, dict) and "blocks" in sec

    def test_always_has_docs_subsection(self, tiny_bertopic_results):
        br = tiny_bertopic_results.drop(columns=["cited_by"])
        cfg = TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5)
        sec = _build_topic_characteristics_section(br, None, cfg, 4)
        sub_titles = [s["title"].lower() for s in _subsections(sec)]
        assert any("doc" in t for t in sub_titles)

    def test_citation_subsections_present_when_cited_by_available(
        self, tiny_bertopic_results
    ):
        cfg = TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5)
        sec = _build_topic_characteristics_section(tiny_bertopic_results, None, cfg, 4)
        assert len(_subsections(sec)) >= 3   # docs + citation + top-N global

    def test_citation_subsections_absent_without_cited_by(self, tiny_bertopic_results):
        br = tiny_bertopic_results.drop(columns=["cited_by"])
        cfg = TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5)
        sec = _build_topic_characteristics_section(br, None, cfg, 4)
        assert len(_subsections(sec)) == 1   # docs only

    def test_uses_topic_labels_in_bar_charts(self, tiny_bertopic_results):
        labels = {0: "TopicA", 1: "TopicB", 2: "TopicC"}
        cfg = TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5)
        sec = _build_topic_characteristics_section(tiny_bertopic_results, labels, cfg, 4)
        # First subsection figure x-axis should use label names
        sub = _subsections(sec)[0]
        fig = _blocks_of_type(sub, "plotly")[0]["figure"]
        x_vals = list(fig.data[0].x)
        assert "TopicA" in x_vals or "TopicB" in x_vals


# =============================================================================
# _build_topic_similarity_section
# =============================================================================

class TestBuildTopicSimilaritySection:

    def test_returns_none_without_topic_cols(self, tiny_bertopic_results):
        br = tiny_bertopic_results.drop(
            columns=[c for c in tiny_bertopic_results.columns if c.startswith("topic#")]
        )
        cfg = TopicSimilarityConfig(clustering=True, dendrogram=True)
        assert _build_topic_similarity_section(br, None, cfg, 5) is None

    def test_returns_section_dict(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=False, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        assert isinstance(sec, dict) and "blocks" in sec

    def test_heatmap_block_always_present(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=False, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        plotly_blocks = _blocks_of_type(sec, "plotly")
        assert plotly_blocks, "Expected a plotly heatmap block"

    def test_dendrogram_block_present_when_enabled(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=True, dendrogram=True)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        plotly_blocks = _blocks_of_type(sec, "plotly")
        # dendrogram + heatmap → at least 2 plotly blocks
        assert len(plotly_blocks) >= 2

    def test_no_dendrogram_block_when_disabled(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=True, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        plotly_blocks = _blocks_of_type(sec, "plotly")
        assert len(plotly_blocks) == 1

    def test_heatmap_diagonal_is_nan(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=False, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        fig  = _blocks_of_type(sec, "plotly")[0]["figure"]
        z    = np.array(fig.data[0].z, dtype=float)
        diag = np.diag(z)
        assert all(np.isnan(v) for v in diag)

    def test_heatmap_off_diagonal_in_range(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=False, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        z = np.array(_blocks_of_type(sec, "plotly")[0]["figure"].data[0].z, dtype=float)
        off_diag = z[~np.isnan(z)]
        assert np.all(off_diag >= -0.01) and np.all(off_diag <= 1.01)

    def test_title_contains_section_number(self, tiny_bertopic_results):
        cfg = TopicSimilarityConfig(clustering=False, dendrogram=False)
        sec = _build_topic_similarity_section(tiny_bertopic_results, None, cfg, 5)
        assert "5." in sec["title"]


# =============================================================================
# _build_paper_selection_section
# =============================================================================

class TestBuildPaperSelectionSection:

    def _run(self, br, ti, *, min_year=2015, proportion=0.5, export=False,
             fmt="csv", labels=None, export_to=None, n=6):
        cfg = PaperSelectionConfig(
            min_year=min_year,
            proportion_per_topic=proportion,
            export_annex=export,
            annex_format=fmt,
        )
        return _build_paper_selection_section(br, ti, labels, cfg, export_to, n)

    def test_returns_none_when_required_columns_missing(self, tiny_bertopic_results,
                                                         tiny_topic_info):
        br = tiny_bertopic_results.drop(columns=["doi"])
        assert self._run(br, tiny_topic_info) is None

    def test_returns_section_dict(self, tiny_bertopic_results, tiny_topic_info):
        sec = self._run(tiny_bertopic_results, tiny_topic_info)
        assert isinstance(sec, dict) and "blocks" in sec

    def test_table_block_present(self, tiny_bertopic_results, tiny_topic_info):
        sec = self._run(tiny_bertopic_results, tiny_topic_info)
        tables = _blocks_of_type(sec, "table")
        assert tables

    def test_table_has_correct_headers(self, tiny_bertopic_results, tiny_topic_info):
        sec = self._run(tiny_bertopic_results, tiny_topic_info)
        headers = _blocks_of_type(sec, "table")[0]["headers"]
        for expected in ("Topic", "Title", "Year", "DOI", "Selection"):
            assert any(expected in h for h in headers), f"Missing header '{expected}'"

    def test_year_filter_excludes_old_papers(self, tiny_bertopic_results, tiny_topic_info):
        # min_year=2023 → only 2023 docs qualify
        sec = self._run(tiny_bertopic_results, tiny_topic_info, min_year=2023)
        rows = _blocks_of_type(sec, "table")[0]["rows"]
        years_in_table = [int(r[4]) for r in rows if r[4].isdigit()]
        assert all(y >= 2023 for y in years_in_table)

    def test_review_papers_detected_by_document_type(self, tiny_bertopic_results,
                                                       tiny_topic_info):
        sec = self._run(tiny_bertopic_results, tiny_topic_info, min_year=2010)
        rows = _blocks_of_type(sec, "table")[0]["rows"]
        selections = [r[-1] for r in rows]
        assert any("Review" in s for s in selections)

    def test_review_papers_detected_by_title(self, tiny_bertopic_results, tiny_topic_info):
        # tiny_bertopic_results has titles containing "Review of..." every 8 docs
        sec = self._run(tiny_bertopic_results, tiny_topic_info, min_year=2010)
        rows = _blocks_of_type(sec, "table")[0]["rows"]
        review_rows = [r for r in rows if "Review" in r[-1]]
        assert review_rows

    def test_annex_csv_created_when_enabled(self, tiny_bertopic_results, tiny_topic_info,
                                             tmp_path):
        self._run(
            tiny_bertopic_results, tiny_topic_info,
            export=True, fmt="csv", export_to=str(tmp_path),
        )
        assert (tmp_path / "paper_selection.csv").exists()

    def test_annex_txt_created_when_enabled(self, tiny_bertopic_results, tiny_topic_info,
                                             tmp_path):
        self._run(
            tiny_bertopic_results, tiny_topic_info,
            export=True, fmt="txt", export_to=str(tmp_path),
        )
        assert (tmp_path / "paper_selection.txt").exists()

    def test_no_annex_file_when_disabled(self, tiny_bertopic_results, tiny_topic_info,
                                          tmp_path):
        self._run(
            tiny_bertopic_results, tiny_topic_info,
            export=False, export_to=str(tmp_path),
        )
        assert not (tmp_path / "paper_selection.csv").exists()

    def test_section_number_in_title(self, tiny_bertopic_results, tiny_topic_info):
        sec = self._run(tiny_bertopic_results, tiny_topic_info, n=6)
        assert "6." in sec["title"]

    def test_topic_labels_appear_in_table(self, tiny_bertopic_results, tiny_topic_info):
        labels = {0: "LabelZero", 1: "LabelOne", 2: "LabelTwo"}
        sec = self._run(tiny_bertopic_results, tiny_topic_info, labels=labels)
        rows = _blocks_of_type(sec, "table")[0]["rows"]
        label_col_values = [r[1] for r in rows]
        assert any("LabelZero" in v or "LabelOne" in v or "LabelTwo" in v
                   for v in label_col_values)


# =============================================================================
# build_report_data — orchestration
# =============================================================================

class TestBuildReportData:

    def _run(self, tiny_best_results, tiny_topic_info, tiny_bertopic_results,
             report_cfg, coupling_dataset=None, topic_labels=None, export_to=None):
        return build_report_data(
            run_dir            = "/fake/run_dir",
            best_model_index        = 0,
            best_results       = tiny_best_results,
            topic_info         = tiny_topic_info,
            bertopic_results   = tiny_bertopic_results,
            report_config      = report_cfg,
            topic_labels       = topic_labels,
            export_to          = export_to,
            coupling_dataset   = coupling_dataset,
        )

    def test_returns_meta_and_sections(self, tiny_best_results, tiny_topic_info,
                                        tiny_bertopic_results, report_cfg):
        rd = self._run(tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg)
        assert "meta" in rd and "sections" in rd

    def test_meta_has_required_keys(self, tiny_best_results, tiny_topic_info,
                                     tiny_bertopic_results, report_cfg):
        meta = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
        )["meta"]
        for key in ("title", "author", "date", "version"):
            assert key in meta

    def test_topics_section_is_first(self, tiny_best_results, tiny_topic_info,
                                      tiny_bertopic_results, report_cfg):
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
        )["sections"]
        assert "Topic" in sections[0]["title"]

    def test_overview_section_is_last(self, tiny_best_results, tiny_topic_info,
                                       tiny_bertopic_results, report_cfg):
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
        )["sections"]
        assert "Technical" in sections[-1]["title"] or "Overview" in sections[-1]["title"]

    def test_networks_section_skipped_without_dataset(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
    ):
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            coupling_dataset=None,
        )["sections"]
        titles = [s["title"] for s in sections]
        assert not any("network" in t.lower() for t in titles)

    def test_networks_section_skipped_when_dataset_has_no_references(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        tmp_path,
    ):
        # No `references` column → no coupling result → no network section.
        p = tmp_path / "no_refs.csv"
        pd.DataFrame({"id": ["a", "b"], "title": ["A", "B"]}).to_csv(p, index=False)
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            coupling_dataset=str(p),
        )["sections"]
        titles = [s["title"] for s in sections]
        assert not any("network" in t.lower() for t in titles)

    def test_actors_section_present_without_references(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        tmp_path,
    ):
        # The actors section reads the `actors` column, not the references: a
        # dataset without references still gets it (and no network section).
        from pysyrev.core.actors import make_actors
        cells = [make_actors([{"key": f"orcid:{a}", "name": a} for a in authors],
                             [{"key": f"ror:{c}", "name": f"U{c}", "country": c}
                              for c in countries])
                 for authors, countries in [(["a", "b"], ["FR", "SN"]),
                                            (["a", "c"], ["FR"]),
                                            (["b", "c"], ["SN", "US"])]]
        p = tmp_path / "actors.csv"
        pd.DataFrame({"id": ["w1", "w2", "w3"], "year": [2019, 2020, 2021],
                      "cited_by": [3, 1, 0], "actors": cells}).to_csv(p, index=False)
        titles = [s["title"] for s in self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            coupling_dataset=str(p))["sections"]]
        assert any("Actors" in t for t in titles)
        assert not any("network" in t.lower() for t in titles)

    def test_actors_section_skipped_without_actors_column(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        tmp_path,
    ):
        p = tmp_path / "no_actors.csv"
        pd.DataFrame({"id": ["a", "b"], "title": ["A", "B"]}).to_csv(p, index=False)
        titles = [s["title"] for s in self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            coupling_dataset=str(p))["sections"]]
        assert not any("Actors" in t for t in titles)

    def test_networks_section_present_with_dataset(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        reviewed_dataset_path,
    ):
        report_cfg.sections.bib_network.coupling.min_size = 3
        report_cfg.sections.bib_network.cocitation.min_size = 3
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            coupling_dataset=reviewed_dataset_path,
        )["sections"]
        titles = [s["title"] for s in sections]
        assert any("network" in t.lower() for t in titles)

    def test_section_numbers_are_sequential(self, tiny_best_results, tiny_topic_info,
                                             tiny_bertopic_results, report_cfg):
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
        )["sections"]
        numbers = []
        for s in sections:
            title = s["title"]
            part  = title.split(".")[0].strip()
            if part.isdigit():
                numbers.append(int(part))
        assert numbers == list(range(1, len(numbers) + 1))

    def test_extra_sections_appended_at_end(self, tiny_best_results, tiny_topic_info,
                                             tiny_bertopic_results, report_cfg):
        from pysyrev.core.config import TopicReportConfig, ReportSectionsConfig
        cfg = TopicReportConfig(
            meta=report_cfg.meta,
            sections=ReportSectionsConfig(
                extra=[{"title": "99. Custom appendix", "blocks": []}]
            ),
        )
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, cfg
        )["sections"]
        assert sections[-1]["title"] == "99. Custom appendix"

    def test_topic_labels_propagate_to_topics_section(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
    ):
        labels = {0: "MyLabel0"}
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            topic_labels=labels,
        )["sections"]
        topics_sec = sections[0]
        # The LLM label appears in the summary table rows or subsection titles
        all_text = str(topics_sec)
        assert "MyLabel0" in all_text


# =============================================================================
# Network table — Title + DOI columns
# =============================================================================

class TestNetworkTableTitleDoi:
    """The top-N-by-strength table must show Title + DOI, not raw node ids.

    Resolution order: corpus by id (documents, in-corpus cited refs by their
    id) → the reference key's own DOI (title looked up when the cited work is in
    corpus) → em dash for an extra-corpus opaque id.
    """

    def _df(self):
        return pd.DataFrame({
            "id":    ["https://openalex.org/W1", "https://openalex.org/W2"],
            "title": ["Justice in energy models", "Grid decarbonization"],
            "doi":   ["https://doi.org/10.1000/AAA", "10.1000/bbb"],
            "year":  [2020, 2021], "cited_by": [5, 3],
        })

    def _table(self, node_ids):
        from pysyrev.core.report.sections.networks import _build_network_subsection
        from pysyrev.core.networks.common import NetworkResult
        n = len(node_ids)
        W = np.ones((n, n)) - np.eye(n)
        res = NetworkResult(node_ids=node_ids, W=W,
                            labels=np.zeros(n, dtype=int),
                            coords=np.zeros((n, 2)), modularity=0.0)
        sub = _build_network_subsection(
            res, self._df(), None, None, title="Co-citation",
            filename_prefix="cc", node_kind="reference", count_label="References",
            k=2, color_mode="community", export_to=None)
        return [b for b in sub["blocks"] if b.get("type") == "table"][0]

    def test_headers_are_title_doi(self):
        tbl = self._table(["https://openalex.org/W1", "https://openalex.org/W2"])
        assert tbl["headers"] == ["Title", "DOI", "Strength", "Community", "Topic"]

    def test_in_corpus_by_id_resolves_title_and_normalized_doi(self):
        tbl = self._table(["https://openalex.org/W1", "https://openalex.org/W2"])
        by_title = {r[0]: r[1] for r in tbl["rows"]}
        assert by_title["Justice in energy models"] == "10.1000/aaa"  # doi.org stripped

    def test_reference_expressed_as_doi_resolves_corpus_title(self):
        # A cited reference given as a DOI that matches an in-corpus work.
        tbl = self._table(["10.1000/bbb", "https://openalex.org/W1"])
        by_doi = {r[1]: r[0] for r in tbl["rows"]}
        assert by_doi["10.1000/bbb"] == "Grid decarbonization"

    def test_extra_corpus_opaque_id_is_dash(self):
        tbl = self._table(["https://openalex.org/W999", "https://openalex.org/W1"])
        row = next(r for r in tbl["rows"] if r[0] == "—")
        assert row[1] == "—"


class TestCommunitySubthemesTable:
    """The coupling subsection lists per-community TF-IDF sub-themes when present."""

    def _subsection(self, terms):
        from pysyrev.core.report.sections.networks import _build_network_subsection
        from pysyrev.core.networks.common import NetworkResult
        ids = [f"W{i}" for i in range(4)]
        W = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                      [0, 0, 0, 1], [0, 0, 1, 0]], float)
        res = NetworkResult(node_ids=ids, W=W, labels=np.array([0, 0, 1, 1]),
                            coords=np.zeros((4, 2)), modularity=0.3, terms=terms)
        df = pd.DataFrame({"id": ids, "title": ["t"] * 4,
                           "doi": [f"10.1/{i}" for i in range(4)], "year": [2020] * 4})
        return _build_network_subsection(
            res, df, None, None, title="Coupling", filename_prefix="c",
            node_kind="document", count_label="Documents", k=2,
            color_mode="community", export_to=None)

    def test_subthemes_table_present_with_terms(self):
        sub = self._subsection({0: ["solar", "photovoltaic"], 1: ["wind", "turbine"]})
        tables = [b for b in sub["blocks"] if b.get("type") == "table"]
        theme = next(t for t in tables if "sub-theme" in t["title"].lower())
        assert theme["headers"] == ["Community", "Docs", "Top terms"]
        by_comm = {r[0]: r[2] for r in theme["rows"]}
        assert "solar" in by_comm["C0"] and "wind" in by_comm["C1"]

    def test_no_subthemes_table_without_terms(self):
        sub = self._subsection({})
        assert not any("sub-theme" in b.get("title", "").lower()
                       for b in sub["blocks"] if b.get("type") == "table")


class TestCommunityLLMLabelColumn:
    """When cluster_labels are supplied, the sub-themes table gains an LLM-label
    column, added alongside the raw TF-IDF terms (not replacing them)."""

    def _subsection(self, terms, cluster_labels):
        from pysyrev.core.report.sections.networks import _build_network_subsection
        from pysyrev.core.networks.common import NetworkResult
        ids = [f"W{i}" for i in range(4)]
        W = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                      [0, 0, 0, 1], [0, 0, 1, 0]], float)
        res = NetworkResult(node_ids=ids, W=W, labels=np.array([0, 0, 1, 1]),
                            coords=np.zeros((4, 2)), modularity=0.3, terms=terms)
        df = pd.DataFrame({"id": ids, "title": ["t"] * 4,
                           "doi": [f"10.1/{i}" for i in range(4)], "year": [2020] * 4})
        return _build_network_subsection(
            res, df, None, None, title="Coupling", filename_prefix="c",
            node_kind="document", count_label="Documents", k=2,
            color_mode="community", export_to=None, cluster_labels=cluster_labels)

    def _theme_table(self, sub):
        tables = [b for b in sub["blocks"] if b.get("type") == "table"]
        return next(t for t in tables if "sub-theme" in t["title"].lower())

    def test_llm_label_column_added_next_to_terms(self):
        sub = self._subsection(
            {0: ["solar", "photovoltaic"], 1: ["wind", "turbine"]},
            {0: "Solar photovoltaic deployment", 1: "Onshore wind integration"})
        tbl = self._theme_table(sub)
        assert tbl["headers"] == ["Community", "Docs", "LLM label", "Top terms"]
        by_comm = {r[0]: r for r in tbl["rows"]}
        assert by_comm["C0"][2] == "Solar photovoltaic deployment"
        assert "solar" in by_comm["C0"][3]        # raw terms still present

    def test_missing_label_falls_back_to_dash(self):
        sub = self._subsection(
            {0: ["solar"], 1: ["wind"]}, {0: "Solar photovoltaic deployment"})
        tbl = self._theme_table(sub)
        by_comm = {r[0]: r for r in tbl["rows"]}
        assert by_comm["C1"][2] == "—"

    def test_no_labels_keeps_three_column_table(self):
        sub = self._subsection({0: ["solar"], 1: ["wind"]}, None)
        tbl = self._theme_table(sub)
        assert tbl["headers"] == ["Community", "Docs", "Top terms"]

    def _legend(self, sub):
        fig = next(b for b in sub["blocks"] if b.get("type") == "plotly")["figure"]
        return [t.name for t in fig.data if t.name and t.name.startswith("C")]

    def test_legend_and_hover_name_the_communities_by_their_label(self):
        sub = self._subsection(
            {0: ["solar"], 1: ["wind"]},
            {0: "Solar photovoltaic deployment", 1: "Onshore wind integration"})
        assert self._legend(sub) == ["C0 · Solar photovoltaic deployment",
                                     "C1 · Onshore wind integration"]
        fig = next(b for b in sub["blocks"] if b.get("type") == "plotly")["figure"]
        c0 = next(t for t in fig.data if t.name and t.name.startswith("C0"))
        assert "Community: C0 · Solar photovoltaic deployment" in c0.hovertext[0]

    def test_legend_keeps_the_id_alone_without_a_label(self):
        assert self._legend(self._subsection({0: ["solar"], 1: ["wind"]},
                                             {0: "Solar"})) == ["C0 · Solar", "C1"]
        assert self._legend(self._subsection({}, None)) == ["C0", "C1"]


class TestCoCitedWorksTable:
    """The works each co-citation community is built on come with their DOI."""

    def test_doi_column(self):
        from pysyrev.core.report.sections.networks import _community_profile_blocks
        from pysyrev.core.networks.common import NetworkResult
        ids = ["10.1000/aaa", "https://openalex.org/W2"]
        res = NetworkResult(node_ids=ids, W=np.array([[0, 1], [1, 0]], float),
                            labels=np.array([0, 0]), coords=np.zeros((2, 2)),
                            modularity=0.0)
        meta = {ids[0]: {"title": "Keyed by its DOI", "year": 2020, "authors": ["A"]},
                ids[1]: {"title": "Resolved DOI", "year": 2021, "authors": ["B"],
                         "doi": "https://doi.org/10.1000/BBB"}}
        works = next(b for b in _community_profile_blocks(res, meta)
                     if b["title"].startswith("Most co-cited works"))
        assert works["headers"][-1] == "DOI"
        assert len(works["col_widths"]) == len(works["headers"])
        assert {r[1]: r[-1] for r in works["rows"]} == {
            "Keyed by its DOI": "10.1000/aaa", "Resolved DOI": "10.1000/bbb"}


class TestCommunityTopicMapping:
    """The community × topic heatmap crosses Leiden communities with BERTopic topics."""

    def _coupling(self, communities):
        from pysyrev.core.networks.common import NetworkResult
        n = len(communities)
        ids = [f"W{i}" for i in range(n)]
        W = np.ones((n, n)) - np.eye(n)
        return NetworkResult(node_ids=ids, W=W, labels=np.array(communities),
                             coords=np.zeros((n, 2)), modularity=0.2), ids

    def test_heatmap_counts_documents_per_community_topic(self):
        from pysyrev.core.report.sections.networks import _build_community_topic_mapping_subsection
        coup, ids = self._coupling([0, 0, 0, 1, 1, 1])
        bt = pd.DataFrame({"id": ids, "Topic": [0, 0, 1, 1, 2, 2]})
        sub = _build_community_topic_mapping_subsection(coup, bt, None, None)
        assert sub["title"] == "Community × topic mapping"
        fig = [b for b in sub["blocks"] if b["type"] == "plotly"][0]["figure"]
        z = np.asarray(fig.data[0].z)
        assert z.shape == (2, 3)                    # 2 communities × 3 topics
        assert z.tolist() == [[2, 1, 0], [0, 1, 2]]
        assert list(fig.data[0].y) == ["C0", "C1"]

    def test_none_without_bertopic(self):
        from pysyrev.core.report.sections.networks import _build_community_topic_mapping_subsection
        coup, _ = self._coupling([0, 0, 1, 1])
        assert _build_community_topic_mapping_subsection(coup, None, None, None) is None

    def test_tail_and_outliers_excluded(self):
        from pysyrev.core.report.sections.networks import _build_community_topic_mapping_subsection
        # community -1 (tail) and topic -1 (outlier) must not appear.
        coup, ids = self._coupling([0, 0, 1, 1, -1])
        bt = pd.DataFrame({"id": ids, "Topic": [0, 1, 0, 1, -1]})
        sub = _build_community_topic_mapping_subsection(coup, bt, None, None)
        fig = [b for b in sub["blocks"] if b["type"] == "plotly"][0]["figure"]
        assert list(fig.data[0].y) == ["C0", "C1"]           # no C-1
        assert all("Topic -1" not in str(x) for x in fig.data[0].x)


class TestCompositeByCommunity:
    """The composite is grounded in Leiden communities, with a per-community list."""

    def _inputs(self):
        from pysyrev.core.networks.common import NetworkResult
        ids = [f"W{i}" for i in range(6)]
        bt = pd.DataFrame({
            "id": ids, "ID": ids,
            "year": [2019, 2020, 2021, 2019, 2020, 2021],
            "cited_by": [80, 50, 30, 60, 40, 20],
            "document_type": ["article"] * 6,
            "title": [f"Paper {i}" for i in range(6)],
            "doi": [f"10.1/{i}" for i in range(6)],
            "Topic": [0, 1, 0, 1, 0, 1],
            "Document": [f"solar wind model {i}" for i in range(6)],
        })
        W = np.ones((6, 6)) - np.eye(6)
        coup = NetworkResult(node_ids=ids, W=W, labels=np.array([0, 0, 0, 1, 1, 1]),
                             coords=np.zeros((6, 2)), modularity=0.3,
                             terms={0: ["solar", "photovoltaic"], 1: ["wind", "turbine"]})
        return bt, coup

    def test_composite_scores_ignore_topic_use_community(self):
        from pysyrev.core.report.sections.papers import _composite_scores
        bt, coup = self._inputs()
        det = _composite_scores(coup, bt)
        assert det                                          # scored
        # No "Topic" column at all → still scored (grouping is the Leiden community).
        det2 = _composite_scores(coup, bt.drop(columns=["Topic"]))
        assert set(det) == set(det2)
        assert all(abs(det[k]["score"] - det2[k]["score"]) < 1e-9 for k in det)

