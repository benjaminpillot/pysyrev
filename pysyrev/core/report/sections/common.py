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


def _docs_per_topic_per_year(bertopic_results: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Document counts per topic and per year, outliers excluded.

    Returns a frame indexed by *every* year between the first and the last of
    the corpus — including the ones a topic published nothing in — with one
    column per topic, or ``None`` when no document carries a usable year. The
    zero-filling is what makes the frame safe to draw a curve through: a year
    missing from the index would otherwise be joined across as if the silence
    never happened.
    """
    if "year" not in bertopic_results.columns:
        return None

    br    = _wo_outliers(bertopic_results)
    years = pd.to_numeric(br["year"], errors="coerce")
    br    = br.assign(year=years).dropna(subset=["year"])
    if br.empty:
        return None

    br = br.assign(year=br["year"].astype(int))
    counts = (
        br.groupby(["year", "Topic"]).size()
        .unstack("Topic", fill_value=0)
        .sort_index()
    )
    full_span = range(int(counts.index.min()), int(counts.index.max()) + 1)
    counts = counts.reindex(full_span, fill_value=0)
    counts.index.name = "year"
    return counts.sort_index(axis=1)


def _median_docs_per_year(counts: pd.DataFrame) -> dict:
    """Each topic's median annual output, keyed by topic id.

    The median is taken over the topic's own active span: from the year of its
    first document — counting the years before a topic existed would drag every
    young topic's median to zero — up to the last year of the corpus, so that
    the silence of a topic which has stopped publishing does count against it.
    """
    last_year = int(counts.index.max())
    medians = {}
    for topic_id in counts.columns:
        col      = counts[topic_id]
        produced = col[col > 0]
        if produced.empty:
            medians[int(topic_id)] = 0.0
            continue
        span = col.loc[int(produced.index.min()):last_year]
        medians[int(topic_id)] = float(span.median())
    return medians


def _median_year_per_topic(counts: pd.DataFrame) -> dict:
    """The year by which each topic had published half of its documents.

    The weighted median of the publication years, taken low: the answer is a
    real year of the corpus rather than the 2015.5 an even document count would
    otherwise produce. It places a topic in time — an old topic and a recent one
    can hold exactly the same number of documents.
    """
    years   = counts.index.to_numpy()
    medians = {}
    for topic_id in counts.columns:
        col = counts[topic_id]
        total = float(col.sum())
        if total <= 0:
            medians[int(topic_id)] = None
            continue
        reached = col.cumsum().to_numpy() >= total / 2.0
        medians[int(topic_id)] = int(years[reached.argmax()])
    return medians


def _short_label(label: str, max_len: int = 42) -> str:
    """*label* trimmed to something a figure legend can carry."""
    label = str(label)
    return label if len(label) <= max_len else label[:max_len - 1].rstrip() + "…"
