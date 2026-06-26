"""
Tests for pysyrev.core.report_data section builders.

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
    BibNetworkReportConfig,
    BibNetworkSectionConfig,
    PaperSelectionConfig,
    TemporalSectionConfig,
    TopicCharacteristicsConfig,
    TopicsSectionConfig,
    TopicSimilarityConfig,
)
from pysyrev.core.report_data import (
    _build_bib_network_section,
    _build_overview_section,
    _build_paper_selection_section,
    _build_temporal_section,
    _build_topic_characteristics_section,
    _build_topic_similarity_section,
    _build_topics_section,
    build_report_data,
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

    def _run(self, topic_info, sections_cfg=None, topic_labels=None, nb_topics=3, n=1):
        cfg = sections_cfg or TopicsSectionConfig(n_repr_docs_per_topic=3)

        class _FakeSections:
            topics = cfg

        return _build_topics_section(topic_info, _FakeSections(), topic_labels, nb_topics, n)

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


# =============================================================================
# _build_bib_network_section
# =============================================================================

class TestBuildBibNetworkSection:

    def test_returns_none_when_both_paths_none(self):
        cfg = BibNetworkReportConfig(coupling_graph=None, cocitation_graph=None)
        assert _build_bib_network_section(cfg, 2) is None

    def test_returns_none_when_files_do_not_exist(self, tmp_path):
        cfg = BibNetworkReportConfig(
            coupling_graph=str(tmp_path / "missing.graphml"),
            cocitation_graph=None,
        )
        assert _build_bib_network_section(cfg, 2) is None

    def test_returns_section_with_valid_graphml(self, coupling_graphml):
        cfg = BibNetworkReportConfig(
            coupling_graph=coupling_graphml, cocitation_graph=None
        )
        sec = _build_bib_network_section(cfg, 2)
        assert sec is not None
        assert "blocks" in sec

    def test_title_contains_section_number(self, coupling_graphml):
        cfg = BibNetworkReportConfig(coupling_graph=coupling_graphml, cocitation_graph=None)
        sec = _build_bib_network_section(cfg, 3)
        assert "3." in sec["title"]

    def test_graph_subsection_contains_key_value_stats(self, coupling_graphml):
        cfg = BibNetworkReportConfig(coupling_graph=coupling_graphml, cocitation_graph=None)
        sec = _build_bib_network_section(cfg, 2)
        kv_blocks = _blocks_of_type(sec, "key_value")
        assert kv_blocks, "Expected at least one key_value block with network stats"
        keys = {item["key"] for item in kv_blocks[0]["items"]}
        assert {"Nodes", "Edges", "Density"}.issubset(keys)

    def test_top_nodes_table_present(self, coupling_graphml):
        cfg = BibNetworkReportConfig(coupling_graph=coupling_graphml, cocitation_graph=None)
        sec = _build_bib_network_section(cfg, 2)
        tables = _blocks_of_type(sec, "table")
        assert tables, "Expected a table listing top connected nodes"

    def test_two_graphs_produce_two_subsections(self, coupling_graphml):
        cfg = BibNetworkReportConfig(
            coupling_graph=coupling_graphml,
            cocitation_graph=coupling_graphml,   # reuse same file for simplicity
        )
        sec = _build_bib_network_section(cfg, 2)
        assert len(_subsections(sec)) == 2


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
        cfg = TemporalSectionConfig(variants=["absolute", "cumulative"])
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        assert len(_subsections(sec)) == 2

    def test_all_four_variants_produce_four_subsections(self, tiny_bertopic_results):
        cfg = TemporalSectionConfig(
            variants=["absolute", "cumulative", "normalized", "weighted"]
        )
        sec = _build_temporal_section(tiny_bertopic_results, 3, cfg, 3)
        assert len(_subsections(sec)) == 4

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
             report_cfg, bib_network_config=None, topic_labels=None, export_to=None):
        return build_report_data(
            run_dir            = "/fake/run_dir",
            best_model_index        = 0,
            best_results       = tiny_best_results,
            topic_info         = tiny_topic_info,
            bertopic_results   = tiny_bertopic_results,
            report_config      = report_cfg,
            bib_network_config = bib_network_config,
            topic_labels       = topic_labels,
            export_to          = export_to,
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

    def test_bib_network_section_skipped_when_config_is_none(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg
    ):
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            bib_network_config=None,
        )["sections"]
        titles = [s["title"] for s in sections]
        assert not any("network" in t.lower() for t in titles)

    def test_bib_network_section_skipped_when_enabled_false(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        coupling_graphml,
    ):
        from pysyrev.core.config import TopicReportConfig, ReportSectionsConfig
        cfg = TopicReportConfig(
            meta=report_cfg.meta,
            sections=ReportSectionsConfig(
                bib_network=BibNetworkSectionConfig(enabled="false"),
            ),
        )
        bn = BibNetworkReportConfig(coupling_graph=coupling_graphml)
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, cfg,
            bib_network_config=bn,
        )["sections"]
        titles = [s["title"] for s in sections]
        assert not any("network" in t.lower() for t in titles)

    def test_bib_network_section_present_when_enabled_auto_with_valid_graph(
        self, tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
        coupling_graphml,
    ):
        bn = BibNetworkReportConfig(coupling_graph=coupling_graphml)
        sections = self._run(
            tiny_best_results, tiny_topic_info, tiny_bertopic_results, report_cfg,
            bib_network_config=bn,
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
