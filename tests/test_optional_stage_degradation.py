"""What the pipeline may lose, and what it must never lose.

A run's expensive work — the corpus, the review, the trained topic model — is on
disk long before the enrichment around it happens. So everything that merely
*names* or *decorates* that work degrades to a warning: an unreachable topic
labeller costs the labels, a section that cannot be built costs that section,
and the run still ends with its outputs.

These tests pin that contract at the three levels it lives at: the labeller
itself (one lost name is not all the names), its callers (a lost label set is
not the run), and the report builder (a lost section is not the PDF).
"""

import pandas as pd
import pytest

from pysyrev.core import llm
from pysyrev.core.llm import labeling
from pysyrev.core.config import TopicLabelerConfig
from pysyrev.core.topic_labels import (labels_are_cacheable,
                                       load_cached_labels, save_labels)


def _labeler_config(**kwargs):
    return TopicLabelerConfig(provider="fake", model_id="m", **kwargs)


class _PartialProvider:
    """Answers every community but the one whose id appears in *fails_for*."""

    def __init__(self, *fails_for):
        self.fails_for = [str(t) for t in fails_for]

    async def call(self, messages, model_id, model_args, schema):
        user = messages[-1]["content"]
        if any(term in user for term in self.fails_for):
            raise RuntimeError("labeler unavailable")
        return {"label": "Some label"}, None


# ── The labeller: one lost name is not all the names ────────────────────────

class TestLabellerDegradesPerItem:

    def test_a_failing_item_leaves_the_others_labelled(self, monkeypatch, capsys):
        monkeypatch.setattr(labeling, "_make_provider",
                            lambda *a, **k: _PartialProvider("wind"))
        out = llm.label_clusters({0: ["solar"], 1: ["wind"]},
                                 _labeler_config(max_retries=1))
        assert set(out) == {0}                      # community 1 simply has no name
        assert "left unlabelled" in capsys.readouterr().out

    def test_an_incomplete_set_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(labeling, "_make_provider",
                            lambda *a, **k: _PartialProvider("wind"))
        out = llm.label_clusters({0: ["solar"], 1: ["wind"]},
                                 _labeler_config(max_retries=1))
        # Caching a half-run would freeze its gaps: the cache exists precisely to
        # stop the labeller running again.
        assert not labels_are_cacheable(out)

    def test_a_complete_set_is_cached(self, monkeypatch):
        monkeypatch.setattr(labeling, "_make_provider",
                            lambda *a, **k: _PartialProvider())
        out = llm.label_clusters({0: ["solar"], 1: ["wind"]}, _labeler_config())
        assert set(out) == {0, 1}
        assert labels_are_cacheable(out)

    def test_a_wholly_unavailable_labeller_yields_no_labels_and_no_cache(
            self, monkeypatch, capsys):
        monkeypatch.setattr(labeling, "_make_provider",
                            lambda *a, **k: _PartialProvider("solar", "wind"))
        out = llm.label_clusters({0: ["solar"], 1: ["wind"]},
                                 _labeler_config(max_retries=1))
        assert out == {}
        assert not labels_are_cacheable(out)
        assert "left unlabelled" in capsys.readouterr().out

    def test_a_cached_plain_dict_is_still_considered_complete(self, tmp_path):
        # What comes back from disk is a plain dict, with no completeness flag:
        # it was only ever written because it was complete.
        save_labels(str(tmp_path), "prefix", {0: "Solar"})
        assert labels_are_cacheable(load_cached_labels(str(tmp_path), "prefix"))


# ── TopicModel: a lost label set is not the run ─────────────────────────────

class TestTopicModelLabellingNeverStopsTheRun:

    def _model(self, run_dir):
        from pysyrev.topic_model import TopicModel
        model = TopicModel.__new__(TopicModel)     # no BERTopic needed here
        model._run_dir        = str(run_dir)
        model.best_model_index = 0
        model.labeler_config  = _labeler_config()
        return model

    def test_an_unreachable_labeller_only_warns(self, monkeypatch, tmp_path, capsys):
        import pysyrev.core.llm as llm_mod

        def _boom(*a, **k):
            raise RuntimeError("topic labeler unavailable")

        monkeypatch.setattr(llm_mod, "label_topics", _boom)
        monkeypatch.setattr(
            "pysyrev.core.report.find_best_results_csv",
            lambda run_dir: (str(tmp_path / "best.csv"), "distance"))
        pd.DataFrame({"hdbscan": ["2_2"], "umap": ["5_3"]}).to_csv(
            tmp_path / "best.csv", index=False)
        tinfo = tmp_path / "topic_info"
        tinfo.mkdir()
        pd.DataFrame({"Topic": [0], "Representation": ["['a']"]}).to_csv(
            tinfo / "hdbscan=2_2_umap=5_3_distance=distance.csv", index=False)

        self._model(tmp_path)._label_selected_model()        # must not raise
        assert "labelling skipped" in capsys.readouterr().out

    def test_a_missing_run_directory_only_warns(self, tmp_path, capsys):
        # Nothing on disk at all: even locating the model must not raise here.
        self._model(tmp_path / "nowhere")._label_selected_model()
        assert "labelling skipped" in capsys.readouterr().out


# ── TopicReport: a lost label set is not the report ─────────────────────────

class TestTopicReportDegradesToNumericNames:

    def _report(self, tmp_path, report_cfg, tiny_best_results, tiny_topic_info,
                tiny_bertopic_results):
        from pysyrev.topic_report import TopicReport
        report = TopicReport(
            run_dir        = str(tmp_path / "run"),
            report_config  = report_cfg,
            labeler_config = _labeler_config(),
        )
        report._metrics_csv_path = str(tmp_path / "best.csv")
        report._distance_name    = "distance"
        report._best_results     = tiny_best_results
        report._topic_info       = tiny_topic_info
        report._bertopic_results = tiny_bertopic_results
        return report

    def test_the_pdf_is_still_written_when_the_labeller_is_down(
            self, monkeypatch, tmp_path, report_cfg, tiny_best_results,
            tiny_topic_info, tiny_bertopic_results, capsys):
        import pysyrev.topic_report as topic_report_mod

        def _boom(*a, **k):
            raise RuntimeError("topic labeler unavailable")

        monkeypatch.setattr(topic_report_mod, "label_topics", _boom)
        report = self._report(tmp_path, report_cfg, tiny_best_results,
                              tiny_topic_info, tiny_bertopic_results)
        out = report.generate_report(str(tmp_path / "report.pdf"))

        assert out == str(tmp_path / "report.pdf")
        assert (tmp_path / "report.pdf").stat().st_size > 0
        assert "topic labels skipped" in capsys.readouterr().out


# ── The report builder: a lost section is not the PDF ───────────────────────

class TestOptionalSectionsAreSkippedNotFatal:

    def test_a_failing_section_is_skipped_with_a_warning(self, capsys):
        from pysyrev.core.report.data import _optional_section

        def _boom():
            raise ValueError("no year column")

        assert _optional_section("temporal", _boom) is None
        assert "temporal section skipped" in capsys.readouterr().out

    def test_the_report_survives_a_failing_section(
            self, monkeypatch, tiny_best_results, tiny_topic_info,
            tiny_bertopic_results, report_cfg, capsys):
        from pysyrev.core.report import build_report_data
        from pysyrev.core.report import data as data_mod

        def _boom(*a, **k):
            raise ValueError("plotting backend unavailable")

        monkeypatch.setattr(data_mod, "_build_temporal_section", _boom)
        rd = build_report_data(
            run_dir          = "/fake/run_dir",
            best_model_index = 0,
            best_results     = tiny_best_results,
            topic_info       = tiny_topic_info,
            bertopic_results = tiny_bertopic_results,
            report_config    = report_cfg,
        )
        titles = [s["title"] for s in rd["sections"]]
        assert not any("temporal" in t.lower() for t in titles)
        assert "Topic" in titles[0]                       # the report still stands
        assert "temporal section skipped" in capsys.readouterr().out
