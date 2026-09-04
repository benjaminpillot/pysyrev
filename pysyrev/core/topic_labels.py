"""
Shared utilities for caching LLM-generated topic and community labels.

Both caches are ``{int id: label}`` JSON files written under the topic-model
run directory; they differ only in where they live and what identifies a run:

- **Topic labels** — ``<run_dir>/topic_labels/<file_prefix>.json``. The
  file_prefix matches the naming convention used for topic_info and
  bertopic_results CSVs (hdbscan=…_umap=…_distance=…), so each cached label
  file is unambiguously tied to one specific model run.

- **Community labels** — ``<run_dir>/cluster_labels/<fingerprint>.json``, keyed
  on a stable hash of the *terms themselves* rather than on model
  hyperparameters. This self-invalidates: if the corpus or the Leiden partition
  changes, the community sub-theme terms change, the hash changes, and the LLM
  re-runs; otherwise the cached labels are reused and no call is made. So the
  labeller runs once, never on every report. Coupling and co-citation share the
  directory without colliding — different terms, different key.
"""

import hashlib
import json
import os
from typing import Optional


def _cache_path(run_dir: str, subdir: str, key: str) -> str:
    return os.path.join(run_dir, subdir, f"{key}.json")


def _load(run_dir: str, subdir: str, key: str) -> Optional[dict]:
    path = _cache_path(run_dir, subdir, key)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _save(run_dir: str, subdir: str, key: str, labels: dict) -> str:
    """Write labels to the cache file. Returns the path written."""
    path = _cache_path(run_dir, subdir, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f,
                  ensure_ascii=False, indent=2)
    return path


def terms_fingerprint(terms: dict) -> str:
    """Stable short hash of a ``{community_id: [terms]}`` mapping.

    Order-independent across communities (keys are sorted) but sensitive to the
    terms of each community, so any change to the partition's sub-themes yields a
    different fingerprint.
    """
    canonical = {str(k): list(terms[k]) for k in sorted(terms)}
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def load_cached_labels(run_dir: str, file_prefix: str) -> Optional[dict]:
    return _load(run_dir, "topic_labels", file_prefix)


def save_labels(run_dir: str, file_prefix: str, labels: dict) -> str:
    return _save(run_dir, "topic_labels", file_prefix, labels)


def load_cached_cluster_labels(run_dir: str, terms: dict) -> Optional[dict]:
    return _load(run_dir, "cluster_labels", terms_fingerprint(terms))


def save_cluster_labels(run_dir: str, terms: dict, labels: dict) -> str:
    return _save(run_dir, "cluster_labels", terms_fingerprint(terms), labels)
