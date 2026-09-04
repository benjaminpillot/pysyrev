"""
Loading a topic-model run, and assembling the report from it.

:func:`build_report_data` is the single entry point: it reads one run\'s
outputs, computes the networks the sections share, and returns the declarative
``report_data`` dict :class:`~pysyrev.core.report.engine.PDFReportEngine`
renders. Each section lives in :mod:`pysyrev.core.report.sections`.
"""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from pysyrev.core.report.sections.common import _FIXED_METRIC_COLS
from pysyrev.core.report.sections.networks import (
    _build_networks_section, _cocitation_metadata,
    _label_cocitation_communities, _label_coupling_communities)
from pysyrev.core.report.sections.papers import _build_paper_selection_section
from pysyrev.core.report.sections.temporal import _build_temporal_section
from pysyrev.core.report.sections.topics import (
    _build_topic_characteristics_section, _build_topic_similarity_section,
    _build_topics_section)


def find_best_results_csv(run_dir: str) -> Tuple[str, str]:
    """Locate the unique *_best_results.csv under run_dir/metrics/.

    Returns (csv_path, distance_name).
    """
    pattern = os.path.join(run_dir, "metrics", "*_best_results.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No best_results CSV found under {run_dir}/metrics/. "
            "Run TopicModel first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple best_results CSVs found under {run_dir}/metrics/: "
            f"{matches}. Expected exactly one."
        )
    path = matches[0]
    distance_name = Path(path).stem.replace("_best_results", "")
    return path, distance_name


def build_file_prefix(hdbscan_params: str, umap_params: str,
                      distance_name: str) -> str:
    return f"hdbscan={hdbscan_params}_umap={umap_params}_distance={distance_name}"


def load_topic_info(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "topic_info", f"{file_prefix}.csv")
    return pd.read_csv(path)


def load_bertopic_results(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "bertopic_results", f"{file_prefix}.csv")
    return pd.read_csv(path)


def _build_overview_section(run_dir, best_model_index, best_results, section_n):
    row = best_results.iloc[best_model_index]
    coherence_col = next(
        (c for c in best_results.columns if c not in _FIXED_METRIC_COLS), None
    )
    kv_items = [
        {"key": "Run directory",         "value": run_dir},
        {"key": "Selected model (rank)", "value": best_model_index},
        {"key": "HDBSCAN parameters",    "value": row["hdbscan"]},
        {"key": "UMAP parameters",       "value": row["umap"]},
        {"key": "Number of topics",      "value": int(row["nb_topics"])},
        {"key": "Number of outliers",    "value": int(row["nb_outliers"])},
        {"key": "Entropy",               "value": f"{row['entropy']:.4f}"},
        {"key": "Diversity",             "value": f"{row['diversity']:.4f}"},
        {"key": "Purity",                "value": f"{row['purity']:.4f}"},
    ]
    if coherence_col:
        kv_items.insert(6, {
            "key":   f"Coherence ({coherence_col})",
            "value": f"{row[coherence_col]:.4f}",
        })
    return {
        "title": f"{section_n}. Topic model — Technical overview",
        "blocks": [{"type": "key_value", "items": kv_items}],
    }


def build_report_data(run_dir: str,
                      best_model_index: int,
                      best_results: pd.DataFrame,
                      topic_info: pd.DataFrame,
                      bertopic_results: pd.DataFrame,
                      report_config,
                      topic_labels: dict = None,
                      export_to: str = None,
                      coupling_dataset: str = None,
                      cluster_labeler_config=None) -> dict:
    """Build the declarative report_data dict consumed by PDFReportEngine."""
    meta = report_config.meta
    sec  = report_config.sections
    nb_topics = int(best_results.iloc[best_model_index]["nb_topics"])

    report_data = {
        "meta": {
            "title":    meta.title,
            "subtitle": meta.subtitle,
            "author":   meta.author,
            "date":     datetime.now().strftime(meta.date_format),
            "version":  meta.version,
            "summary":  meta.summary,
        },
        "sections": [],
    }

    n = 1  # running section counter

    # 1. Topics
    report_data["sections"].append(
        _build_topics_section(topic_info, sec, topic_labels, nb_topics, n)
    )
    n += 1

    # Compute the coupling and co-citation networks once from the reviewed
    # dataset's raw references — shared by the networks section and by
    # network-based paper selection.
    _networks_df = _coupling_result = _cocitation_result = None
    if not coupling_dataset:
        print("[report] networks skipped: no coupling dataset was provided "
              "(topic_model.doc_dataset is empty/unresolved) — the networks, "
              "connectivity and composite sections will be absent.")
    if coupling_dataset:
        try:
            _networks_df = pd.read_csv(coupling_dataset, low_memory=False)
        except Exception as e:
            print(f"[report] networks skipped: cannot read {coupling_dataset!r} ({e!r})")
            _networks_df = None
        if _networks_df is not None and not {"references", "id"}.issubset(_networks_df.columns):
            print(f"[report] networks skipped: dataset has no 'references'/'id' "
                  f"columns (found {list(_networks_df.columns)}).")
            _networks_df = None
        if _networks_df is not None:
            from pysyrev.core.networks import build_coupling, build_cocitation
            nc = sec.bib_network
            # Prefer the canonical DOI-keyed references when available: they
            # unify references across merged sources (see reference_keys). Fall
            # back to the raw references column otherwise.
            ref_col = ("reference_keys"
                       if "reference_keys" in _networks_df.columns
                       else "references")
            n_with_refs = int(_networks_df[ref_col].apply(
                lambda v: isinstance(v, str) and bool(v.strip())).sum())
            if n_with_refs < 2:
                print(f"[report] networks likely empty: only {n_with_refs} document(s) "
                      f"carry references in column {ref_col!r}.")
            try:
                _coupling_result = build_coupling(
                    _networks_df, ref_col=ref_col,
                    resolution_range=nc.coupling.resolution_range,
                    resolution_step=nc.coupling.resolution_step,
                    min_size=nc.coupling.min_size)
            except Exception as e:
                print(f"[report] coupling network skipped: {e!r}")
                _coupling_result = None
            try:
                _cocitation_result = build_cocitation(
                    _networks_df, ref_col=ref_col, min_ref_freq=nc.cocitation.min_ref_freq,
                    resolution_range=nc.cocitation.resolution_range,
                    resolution_step=nc.cocitation.resolution_step,
                    min_size=nc.cocitation.min_size,
                    ref_meta=_cocitation_metadata(_networks_df, ref_col, nc.cocitation))
            except Exception as e:
                print(f"[report] co-citation network skipped: {e!r}")
                _cocitation_result = None

    # Human-readable labels for the communities of both networks, via the same
    # LLM used for topics. Computed once and cached (keyed on a hash of the
    # community sub-theme terms), so the labeller never re-runs on an unchanged
    # partition. Co-citation only has terms when its metadata step ran.
    _cluster_labels = _label_coupling_communities(
        run_dir, _coupling_result, _networks_df, cluster_labeler_config)
    _cocitation_labels = _label_cocitation_communities(
        run_dir, _cocitation_result, cluster_labeler_config)

    # 2. Bibliographic networks (coupling + co-citation) — rendered whenever the
    # reviewed dataset yielded a coupling network (i.e. it carried references).
    if _coupling_result is not None:
        section = _build_networks_section(
            _networks_df, _coupling_result, _cocitation_result,
            bertopic_results, topic_labels, sec.bib_network, export_to, n,
            cluster_labels=_cluster_labels,
            cocitation_labels=_cocitation_labels)
        if section is not None:
            report_data["sections"].append(section)
            n += 1

    # 3. Temporal dynamics
    section = _build_temporal_section(bertopic_results, nb_topics, sec.temporal, n)
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 4. Topic characteristics
    section = _build_topic_characteristics_section(
        bertopic_results, topic_labels, sec.topic_characteristics, n
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 5. Topic similarity
    section = _build_topic_similarity_section(
        bertopic_results, topic_labels, sec.topic_similarity, n
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 6. Paper selection
    section = _build_paper_selection_section(
        bertopic_results, topic_info, topic_labels, sec.paper_selection, export_to, n,
        coupling_result=_coupling_result,
        cocitation_result=_cocitation_result,
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 7. Technical appendix
    report_data["sections"].append(
        _build_overview_section(run_dir, best_model_index, best_results, n)
    )

    # Extra user-defined sections
    if sec.extra:
        for extra_section in sec.extra:
            report_data["sections"].append(extra_section)

    return report_data
