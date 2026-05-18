"""
Data-loading and report-building helpers for the topic-report pipeline.

Called by TopicReport (report_data.py). Mirrors the split between
core/topic.py and topic_model.py: core contains the pure functions,
the runtime class coordinates them and holds instance-level cache.
"""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd


_FIXED_METRIC_COLS = frozenset({
    "hdbscan", "umap", "nb_topics", "entropy", "diversity",
    "nb_outliers", "distance", "model", "purity", "distance_with_purity",
})


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
    """Build the filename stem shared by topic_info and bertopic_results CSVs."""
    return f"hdbscan={hdbscan_params}_umap={umap_params}_distance={distance_name}"


def load_topic_info(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "topic_info", f"{file_prefix}.csv")
    return pd.read_csv(path)


def load_bertopic_results(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "bertopic_results", f"{file_prefix}.csv")
    return pd.read_csv(path)


def build_report_data(run_dir: str,
                      model_index: int,
                      best_results: pd.DataFrame,
                      topic_info: pd.DataFrame,
                      report_config,
                      topic_labels: dict = None) -> dict:
    """Build the declarative report_data dict consumed by PDFReportEngine."""
    meta = report_config.meta
    row  = best_results.iloc[model_index]

    coherence_col = next(
        (c for c in best_results.columns if c not in _FIXED_METRIC_COLS), None
    )

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

    # Section 1: model overview
    kv_items = [
        {"key": "Run directory",         "value": run_dir},
        {"key": "Selected model (rank)", "value": model_index},
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

    report_data["sections"].append({
        "title": "1. Topic model — Overview",
        "blocks": [{"type": "key_value", "items": kv_items}],
    })

    # Section 2: topics table
    nb_topics  = int(row["nb_topics"])
    use_labels = topic_labels is not None
    headers    = ["ID", "LLM label" if use_labels else "Name", "# Docs", "Top words"]
    topic_rows = []
    for _, t in topic_info.iterrows():
        topic_id = int(t.get("Topic", -1))
        if topic_id == -1:
            continue
        if use_labels:
            name = topic_labels.get(topic_id, str(t.get("Name", "-")))
        else:
            name = str(t.get("Name", "-"))
        topic_rows.append([
            str(topic_id),
            name,
            str(int(t.get("Count", 0))),
            str(t.get("Representation", "-")),
        ])

    report_data["sections"].append({
        "title": "2. Topics",
        "blocks": [
            {
                "type":    "table",
                "title":   f"{nb_topics} topics identified",
                "headers": headers,
                "rows":    topic_rows,
            },
        ],
    })

    # Optional user-defined sections from the YAML
    if report_config.sections:
        for section in report_config.sections:
            report_data["sections"].append(section)

    return report_data
