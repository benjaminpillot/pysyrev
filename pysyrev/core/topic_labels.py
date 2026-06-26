"""
Shared utilities for caching LLM-generated topic labels.

Labels are stored as JSON files alongside the other topic-model outputs:
    <run_dir>/topic_labels/<file_prefix>.json

The file_prefix matches the naming convention used for topic_info and
bertopic_results CSVs (hdbscan=…_umap=…_distance=…), so each cached
label file is unambiguously tied to one specific model run.
"""

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
