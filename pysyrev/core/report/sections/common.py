"""Small helpers shared by more than one report section."""

import ast
from typing import Optional

import pandas as pd


_FIXED_METRIC_COLS = frozenset({
    "hdbscan", "umap", "nb_topics", "entropy", "diversity",
    "nb_outliers", "distance", "model", "purity", "distance_with_purity",
})


def _safe_parse(val):
    """Parse a stringified Python list stored in a CSV cell."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _topic_label(topic_id: int, topic_labels: Optional[dict]) -> str:
    if topic_labels and topic_id in topic_labels:
        return topic_labels[topic_id]
    return f"Topic {topic_id}"


def _wo_outliers(bertopic_results: pd.DataFrame) -> pd.DataFrame:
    return bertopic_results[bertopic_results["Topic"] != -1].copy()


def _topic_cols(bertopic_results: pd.DataFrame):
    return sorted(
        [c for c in bertopic_results.columns if c.startswith("topic#")],
        key=lambda c: int(c.split("#")[1])
    )
