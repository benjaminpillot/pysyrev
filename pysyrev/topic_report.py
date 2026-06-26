"""
Runtime model for the topic-report pipeline.

TopicReport reads the pre-exported outputs of a TopicModel run and builds a
PDF report for the selected model. It mirrors TopicModel/LLMReview: pure
runtime, no YAML knowledge, built from parsed configs via `from_config`.

Business logic lives in pysyrev/core/report_data.py. This class only
coordinates the core functions and holds instance-level lazy-loaded state.

Typical usage
-------------
    cfg    = Config.load("config.yaml")
    report = TopicReport.from_config(cfg)
    report.generate_report()
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from pysyrev.core.config import Config
from pysyrev.core.llm import label_topics
from pysyrev.core.report import PDFReportEngine
from pysyrev.core.report_data import (
    find_best_results_csv,
    build_file_prefix,
    load_topic_info,
    load_bertopic_results,
    build_report_data,
)
from pysyrev.core.topic_labels import load_cached_labels, save_labels


@dataclass
class TopicReport:

    run_dir:        str
    report_config:  object           # ReportConfig
    best_model_index:    int              = 0
    export_to:      Union[None, str] = None
    labeler_config: object           = None  # TopicLabelerConfig or None

    _best_results:     pd.DataFrame = field(default=None, init=False, repr=False)
    _topic_info:       pd.DataFrame = field(default=None, init=False, repr=False)
    _bertopic_results: pd.DataFrame = field(default=None, init=False, repr=False)
    _metrics_csv_path: str          = field(default=None, init=False, repr=False)
    _distance_name:    str          = field(default=None, init=False, repr=False)

    # ---- bridge from configuration ----------------------------------------

    bib_network_config: object           = None  # BibNetworkReportConfig or None

    @classmethod
    def from_config(cls, config: Config) -> 'TopicReport':
        if config.topic_report is None:
            raise ValueError(
                "Config has no 'topic_report' section. "
                "Add it to your YAML to enable report generation."
            )
        return cls(
            run_dir            = config.topic_report.run_dir,
            report_config      = config.topic_report,
            best_model_index        = config.topic_model.best_model_index if config.topic_model is not None else 0,
            export_to          = config.topic_report.export_to,
            labeler_config     = config.llm,
            bib_network_config = config.bib_network_graphs,
        )

    # ---- internals --------------------------------------------------------

    def _resolve_metrics_csv(self):
        """Locate best_results CSV once; cache path and distance name."""
        if self._metrics_csv_path is not None:
            return
        self._metrics_csv_path, self._distance_name = find_best_results_csv(
            self.run_dir
        )

    def _file_prefix(self) -> str:
        self._resolve_metrics_csv()
        row = self.selected_model_row
        return build_file_prefix(row["hdbscan"], row["umap"], self._distance_name)

    # ---- public properties ------------------------------------------------

    @property
    def best_results(self) -> pd.DataFrame:
        if self._best_results is None:
            self._resolve_metrics_csv()
            self._best_results = pd.read_csv(self._metrics_csv_path)
        return self._best_results

    @property
    def selected_model_row(self) -> pd.Series:
        n = len(self.best_results)
        if self.best_model_index >= n:
            raise IndexError(
                f"best_model_index={self.best_model_index} out of range "
                f"(only {n} result(s) available in best_results)."
            )
        return self.best_results.iloc[self.best_model_index]

    @property
    def topic_info(self) -> pd.DataFrame:
        if self._topic_info is None:
            self._topic_info = load_topic_info(self.run_dir, self._file_prefix())
        return self._topic_info

    @property
    def bertopic_results(self) -> pd.DataFrame:
        if self._bertopic_results is None:
            self._bertopic_results = load_bertopic_results(
                self.run_dir, self._file_prefix()
            )
        return self._bertopic_results

    @property
    def nb_topics(self) -> int:
        return int(self.selected_model_row["nb_topics"])

    @property
    def metrics(self) -> pd.Series:
        return self.selected_model_row

    # ---- public API -------------------------------------------------------

    def generate_report(self, output_file: Union[None, str] = None) -> str:
        """Generate the PDF report. Returns the path of the written file."""
        if output_file is None:
            if self.export_to is None:
                raise ValueError(
                    "Provide output_file or set export_to in the config."
                )
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_subdir = os.path.join(self.export_to, f"report_{ts}")
            output_file = os.path.join(run_subdir, f"topic_report_{ts}.pdf")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        topic_labels = None
        if self.labeler_config is not None:
            file_prefix = self._file_prefix()
            cached = load_cached_labels(self.run_dir, file_prefix)
            if cached is not None:
                topic_labels = cached
            else:
                print("Generating human-readable topic labels via LLM…")
                topic_labels = label_topics(self.topic_info, self.labeler_config)
                path = save_labels(self.run_dir, file_prefix, topic_labels)
                print(f"Topic labels saved to: {path}")

        report_data = build_report_data(
            self.run_dir,
            self.best_model_index,
            self.best_results,
            self.topic_info,
            self.bertopic_results,
            self.report_config,
            bib_network_config = self.bib_network_config,
            topic_labels       = topic_labels,
            export_to          = str(Path(output_file).parent),
        )
        PDFReportEngine().generate(report_data, output_file)
        return output_file
