"""End-to-end integration tests: bib_network → topic_model → topic_report.

Run with:
    pytest -m integration

All tests are skipped by default to keep the default test suite fast.
The session-scoped fixtures that do the heavy ML work live in conftest.py.
"""

import ast
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from pysyrev.core.config import (
    BibNetworkReportConfig,
    BibNetworkSectionConfig,
    PaperSelectionConfig,
    ReportMetaConfig,
    ReportSectionsConfig,
    TemporalSectionConfig,
    TopicCharacteristicsConfig,
    TopicReportConfig,
    TopicsSectionConfig,
    TopicSimilarityConfig,
)
from pysyrev.topic_report import TopicReport


# ── Shared helpers ───────────────────────────────────────────────────────────

def _best_results_csv(run_dir: Path) -> Path:
    return next((run_dir / "metrics").glob("*_best_results.csv"))


def _topic_info_csv(run_dir: Path) -> Path:
    return next((run_dir / "topic_info").glob("*.csv"))


def _bertopic_results_csv(run_dir: Path) -> Path:
    return next((run_dir / "bertopic_results").glob("*.csv"))


def _minimal_report_config() -> TopicReportConfig:
    return TopicReportConfig(
        meta=ReportMetaConfig(
            title="Integration test report",
            author="pytest",
            date_format="%Y",
            version="0.0",
        ),
        sections=ReportSectionsConfig(
            topics=TopicsSectionConfig(n_repr_docs_per_topic=2),
            bib_network=BibNetworkSectionConfig(enabled="auto"),
            temporal=TemporalSectionConfig(variants=["absolute"]),
            topic_characteristics=TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5),
            topic_similarity=TopicSimilarityConfig(clustering=False, dendrogram=False),
            paper_selection=PaperSelectionConfig(
                min_year=2020,
                proportion_per_topic=0.5,
                export_annex=True,
                annex_format="csv",
            ),
        ),
    )


# ── BibNetwork ───────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestBibNetworkIntegration:

    def test_coupling_graph_has_nodes(self, bib_network_outputs):
        G = bib_network_outputs["net"].coupling_graph
        assert G.number_of_nodes() > 0

    def test_coupling_graph_has_edges(self, bib_network_outputs):
        G = bib_network_outputs["net"].coupling_graph
        assert G.number_of_edges() > 0

    def test_cocitation_graph_has_nodes(self, bib_network_outputs):
        G = bib_network_outputs["net"].cocitation_graph
        assert G.number_of_nodes() > 0

    def test_graphml_files_exist(self, bib_network_outputs):
        assert Path(bib_network_outputs["coupling_graphml"]).exists()
        assert Path(bib_network_outputs["cocitation_graphml"]).exists()

    def test_coupling_graphml_is_loadable(self, bib_network_outputs):
        G = nx.read_graphml(bib_network_outputs["coupling_graphml"])
        assert G.number_of_nodes() > 0

    def test_cocitation_graphml_is_loadable(self, bib_network_outputs):
        G = nx.read_graphml(bib_network_outputs["cocitation_graphml"])
        assert G.number_of_nodes() > 0

    def test_coupling_nodes_have_title_attr(self, bib_network_outputs):
        G = bib_network_outputs["net"].coupling_graph
        node_data = dict(list(G.nodes(data=True))[:1])
        first = next(iter(node_data.values()))
        assert "title" in first

    # ── save_vosviewer ───────────────────────────────────────────────────────

    def test_vosviewer_files_created(self, bib_network_outputs, tmp_path):
        from pysyrev.core.config import BibNetworkExportConfig
        cfg = BibNetworkExportConfig(export_dir=str(tmp_path), run_name="vos_test")
        bib_network_outputs["net"].save_vosviewer(cfg)
        assert Path(cfg.citation_vos).exists()
        assert Path(cfg.coupling_vos).exists()
        assert Path(cfg.cocitation_vos).exists()

    def test_vosviewer_json_structure(self, bib_network_outputs, tmp_path):
        import json
        from pysyrev.core.config import BibNetworkExportConfig
        cfg = BibNetworkExportConfig(export_dir=str(tmp_path), run_name="vos_struct")
        bib_network_outputs["net"].save_vosviewer(cfg)

        data = json.loads(Path(cfg.coupling_vos).read_text())
        net  = data["network"]
        assert "items" in net and "links" in net and "clusters" in net
        assert len(net["items"]) > 0
        first = net["items"][0]
        assert {"id", "label"} <= first.keys()

    def test_vosviewer_with_topic_map(self, bib_network_outputs, tmp_path):
        import json
        from pysyrev.core.config import BibNetworkExportConfig

        net = bib_network_outputs["net"]
        node_ids = list(net.coupling_graph.nodes())
        topic_map = {nid: (i % 3) for i, nid in enumerate(node_ids)}
        topic_label_map = {0: "Topic A", 1: "Topic B", 2: "Topic C"}

        cfg = BibNetworkExportConfig(export_dir=str(tmp_path), run_name="vos_topics")
        net.save_vosviewer(cfg, topic_map=topic_map, topic_label_map=topic_label_map)

        data    = json.loads(Path(cfg.coupling_vos).read_text())
        items   = data["network"]["items"]
        clusters = data["network"]["clusters"]

        assert any("cluster" in it for it in items)
        assert len(clusters) > 0
        cluster_labels = {c["label"] for c in clusters}
        assert cluster_labels <= {"Topic A", "Topic B", "Topic C"}

    def test_vosviewer_topic_in_description(self, bib_network_outputs, tmp_path):
        import json
        from pysyrev.core.config import BibNetworkExportConfig

        net = bib_network_outputs["net"]
        node_ids = list(net.coupling_graph.nodes())
        topic_map = {nid: 0 for nid in node_ids}
        topic_label_map = {0: "Agent-based models"}

        cfg = BibNetworkExportConfig(export_dir=str(tmp_path), run_name="vos_desc")
        net.save_vosviewer(cfg, topic_map=topic_map, topic_label_map=topic_label_map)

        data  = json.loads(Path(cfg.coupling_vos).read_text())
        items = data["network"]["items"]
        assert all(
            "Topic: Agent-based models" in it.get("description", "")
            for it in items if "cluster" in it
        )

    def test_vosviewer_cocitation_topic_mapping(self, bib_network_outputs, tmp_path):
        """Topic map (plain paper IDs) is correctly remapped for co-citation R:-prefixed nodes."""
        import json
        from pysyrev.core.config import BibNetworkExportConfig

        net = bib_network_outputs["net"]
        # Build topic_map from plain IDs (as exposed to the user from bertopic results)
        internal_ids = [
            n[2:] for n in net.cocitation_graph.nodes()
            if n.startswith("R:")
        ]
        topic_map = {pid: 1 for pid in internal_ids}
        topic_label_map = {1: "Main topic"}

        cfg = BibNetworkExportConfig(export_dir=str(tmp_path), run_name="vos_cocit")
        net.save_vosviewer(cfg, topic_map=topic_map, topic_label_map=topic_label_map)

        data  = json.loads(Path(cfg.cocitation_vos).read_text())
        items = data["network"]["items"]
        assert any("cluster" in it for it in items)


# ── TopicModel ───────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestTopicModelIntegration:

    def test_output_directories_exist(self, topic_model_outputs):
        run_dir = Path(topic_model_outputs["run_dir"])
        assert (run_dir / "metrics").is_dir()
        assert (run_dir / "topic_info").is_dir()
        assert (run_dir / "bertopic_results").is_dir()

    def test_best_results_csv_exists(self, topic_model_outputs):
        run_dir = Path(topic_model_outputs["run_dir"])
        csvs = list((run_dir / "metrics").glob("*_best_results.csv"))
        assert len(csvs) == 1

    def test_best_results_has_required_columns(self, topic_model_outputs):
        df = pd.read_csv(_best_results_csv(Path(topic_model_outputs["run_dir"])))
        for col in ("hdbscan", "umap", "nb_topics", "entropy", "diversity", "nb_outliers"):
            assert col in df.columns, f"Missing column in best_results: {col}"

    def test_best_results_row_count_within_keep_n(self, topic_model_outputs):
        df = pd.read_csv(_best_results_csv(Path(topic_model_outputs["run_dir"])))
        assert len(df) <= 3   # keep_n_results=3

    def test_bertopic_results_has_metadata_columns(self, topic_model_outputs):
        df = pd.read_csv(_bertopic_results_csv(Path(topic_model_outputs["run_dir"])))
        for col in ("year", "cited_by", "document_type", "title", "doi"):
            assert col in df.columns, f"Missing metadata column: {col}"

    def test_bertopic_results_has_topic_probability_columns(self, topic_model_outputs):
        df = pd.read_csv(_bertopic_results_csv(Path(topic_model_outputs["run_dir"])))
        topic_cols = [c for c in df.columns if c.startswith("topic#")]
        assert len(topic_cols) > 0

    def test_bertopic_results_document_count(self, topic_model_outputs):
        df = pd.read_csv(_bertopic_results_csv(Path(topic_model_outputs["run_dir"])))
        assert len(df) > 0

    def test_topic_info_has_repr_doc_columns(self, topic_model_outputs):
        df = pd.read_csv(_topic_info_csv(Path(topic_model_outputs["run_dir"])))
        for col in ("repr_doc_title", "repr_doc_year", "repr_doc_cited_by"):
            assert col in df.columns, f"Missing repr_doc column: {col}"

    def test_topic_info_repr_doc_title_is_parseable_list(self, topic_model_outputs):
        df = pd.read_csv(_topic_info_csv(Path(topic_model_outputs["run_dir"])))
        for raw in df["repr_doc_title"].dropna():
            parsed = ast.literal_eval(raw)
            assert isinstance(parsed, list)
            assert all(isinstance(s, str) for s in parsed)

    def test_topic_info_repr_doc_year_is_parseable_list(self, topic_model_outputs):
        df = pd.read_csv(_topic_info_csv(Path(topic_model_outputs["run_dir"])))
        for raw in df["repr_doc_year"].dropna():
            parsed = ast.literal_eval(raw)
            assert isinstance(parsed, list)


# ── TopicReport ──────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestTopicReportIntegration:

    def test_pdf_is_generated(self, tmp_path, topic_model_outputs):
        report = TopicReport(
            run_dir=topic_model_outputs["run_dir"],
            report_config=_minimal_report_config(),
            model_index=0,
            export_to=str(tmp_path),
        )
        out = str(tmp_path / "test_report.pdf")
        result = report.generate_report(output_file=out)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 1024   # non-trivial PDF

    def test_paper_selection_annex_csv_is_created(self, tmp_path, topic_model_outputs):
        report = TopicReport(
            run_dir=topic_model_outputs["run_dir"],
            report_config=_minimal_report_config(),
            model_index=0,
            export_to=str(tmp_path),
        )
        report.generate_report(output_file=str(tmp_path / "test_report.pdf"))
        assert (tmp_path / "paper_selection.csv").exists()

    def test_paper_selection_annex_csv_has_expected_columns(self, tmp_path, topic_model_outputs):
        report = TopicReport(
            run_dir=topic_model_outputs["run_dir"],
            report_config=_minimal_report_config(),
            model_index=0,
            export_to=str(tmp_path),
        )
        report.generate_report(output_file=str(tmp_path / "test_report.pdf"))
        annex = pd.read_csv(tmp_path / "paper_selection.csv")
        for col in ("title", "year", "doi"):
            assert col in annex.columns, f"Missing column in annex: {col}"

    def test_report_with_explicit_bib_network_paths(
        self, tmp_path, topic_model_outputs, bib_network_outputs
    ):
        bib_cfg = BibNetworkReportConfig(
            coupling_graph=bib_network_outputs["coupling_graphml"],
            cocitation_graph=bib_network_outputs["cocitation_graphml"],
        )
        report = TopicReport(
            run_dir=topic_model_outputs["run_dir"],
            report_config=_minimal_report_config(),
            model_index=0,
            export_to=str(tmp_path),
            bib_network_config=bib_cfg,
        )
        out = str(tmp_path / "test_report_bib.pdf")
        result = report.generate_report(output_file=out)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 1024

    def test_no_annex_when_disabled(self, tmp_path, topic_model_outputs):
        cfg = TopicReportConfig(
            meta=ReportMetaConfig(
                title="No-annex report",
                author="pytest",
                date_format="%Y",
                version="0.0",
            ),
            sections=ReportSectionsConfig(
                topics=TopicsSectionConfig(n_repr_docs_per_topic=2),
                temporal=TemporalSectionConfig(variants=["absolute"]),
                topic_characteristics=TopicCharacteristicsConfig(n_top_cited_per_topic=2, n_top_cited_global=5),
                topic_similarity=TopicSimilarityConfig(clustering=False, dendrogram=False),
                paper_selection=PaperSelectionConfig(
                    min_year=2020,
                    proportion_per_topic=0.5,
                    export_annex=False,
                ),
            ),
        )
        report = TopicReport(
            run_dir=topic_model_outputs["run_dir"],
            report_config=cfg,
            model_index=0,
            export_to=str(tmp_path),
        )
        report.generate_report(output_file=str(tmp_path / "test_report.pdf"))
        assert not (tmp_path / "paper_selection.csv").exists()
        assert not (tmp_path / "paper_selection.txt").exists()
