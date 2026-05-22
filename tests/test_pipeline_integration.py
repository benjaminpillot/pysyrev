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
    ReportConfig,
    ReportMetaConfig,
    ReportSectionsConfig,
    TemporalSectionConfig,
    TopicCharacteristicsConfig,
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


def _minimal_report_config() -> ReportConfig:
    return ReportConfig(
        meta=ReportMetaConfig(
            title="Integration test report",
            author="pytest",
            date_format="%Y",
            version="0.0",
        ),
        sections=ReportSectionsConfig(
            topics=TopicsSectionConfig(nr_repr_docs=2),
            bib_network=BibNetworkSectionConfig(enabled="auto"),
            temporal=TemporalSectionConfig(variants=["absolute"]),
            topic_characteristics=TopicCharacteristicsConfig(top_n_per_topic=2, top_n_global=5),
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
        cfg = ReportConfig(
            meta=ReportMetaConfig(
                title="No-annex report",
                author="pytest",
                date_format="%Y",
                version="0.0",
            ),
            sections=ReportSectionsConfig(
                topics=TopicsSectionConfig(nr_repr_docs=2),
                temporal=TemporalSectionConfig(variants=["absolute"]),
                topic_characteristics=TopicCharacteristicsConfig(top_n_per_topic=2, top_n_global=5),
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
