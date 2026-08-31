"""
Shared utilities for caching LLM-generated topic labels.

Labels are stored as JSON files alongside the other topic-model outputs:
    <run_dir>/topic_labels/<file_prefix>.json

The file_prefix matches the naming convention used for topic_info and
bertopic_results CSVs (hdbscan=…_umap=…_distance=…), so each cached
label file is unambiguously tied to one specific model run.
"""

import hashlib
import json
import os
from typing import Optional


def labels_cache_path(run_dir: str, file_prefix: str) -> str:
    return os.path.join(run_dir, "topic_labels", f"{file_prefix}.json")


def load_cached_labels(run_dir: str, file_prefix: str) -> Optional[dict]:
    path = labels_cache_path(run_dir, file_prefix)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_labels(run_dir: str, file_prefix: str, labels: dict) -> str:
    """Write labels to the cache file. Returns the path written."""
    path = labels_cache_path(run_dir, file_prefix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f,
                  ensure_ascii=False, indent=2)
    return path


# ── Coupling-community labels ────────────────────────────────────────────────
#
# Community labels are cached alongside the topic labels, but keyed on a stable
# hash of the *terms themselves* (``<run_dir>/cluster_labels/<hash>.json``)
# rather than on model hyperparameters. This self-invalidates: if the corpus or
# the Leiden partition changes, the community sub-theme terms change, the hash
# changes, and the LLM re-runs; otherwise the cached labels are reused and no
# call is made. So the labeller runs once, never on every report.


def terms_fingerprint(terms: dict) -> str:
    """Stable short hash of a ``{community_id: [terms]}`` mapping.

    Order-independent across communities (keys are sorted) but sensitive to the
    terms of each community, so any change to the partition's sub-themes yields a
    different fingerprint.
    """
    canonical = {str(k): list(terms[k]) for k in sorted(terms)}
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def cluster_labels_cache_path(run_dir: str, fingerprint: str) -> str:
    return os.path.join(run_dir, "cluster_labels", f"{fingerprint}.json")


def load_cached_cluster_labels(run_dir: str, terms: dict) -> Optional[dict]:
    path = cluster_labels_cache_path(run_dir, terms_fingerprint(terms))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_cluster_labels(run_dir: str, terms: dict, labels: dict) -> str:
    """Write community labels to their fingerprint-keyed cache file."""
    path = cluster_labels_cache_path(run_dir, terms_fingerprint(terms))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f,
                  ensure_ascii=False, indent=2)
    return path
