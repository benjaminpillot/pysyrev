"""
merge_bibs — hybrid version with n-gram blocking for fuzzy matching.

Four-pass strategy, from cheapest to most expensive:
  1. Normalized DOIs (lowercase + strip prefixes) -> exact match via hashing.
  2. Canonicalized titles (lowercase + accents + punctuation) -> exact match.
  3. N-gram blocking: build an inverted index of word 3-grams and shortlist,
     per unmatched row of main, a few dozen candidates from other.
  4. Fuzzy matching (rapidfuzz) on each row's shortlist only.

Compared to doing `rapidfuzz.cdist(main, other)` blindly, blocking turns an
O(N*M) comparison into O(N*k) where k is typically ~30-100 candidates per
row, independent of M. On 9k x 9k inputs, this is usually 30-100x faster.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process


# =============================================================================
# Normalization
# =============================================================================

_DOI_PREFIXES = re.compile(
    r'^\s*(?:https?://)?(?:dx\.)?doi\.org/', re.IGNORECASE
)


def _normalize_doi(series):
    """Canonicalize DOIs: lowercase, strip http(s)://[dx.]doi.org/, strip whitespace."""
    s = series.fillna('').astype(str).str.strip().str.lower()
    s = s.str.replace(_DOI_PREFIXES, '', regex=True)
    return s


def _canonicalize_title(series):
    """Canonicalize titles: lowercase, strip accents, strip punctuation,
    normalize whitespace."""
    s = series.fillna('').astype(str).str.lower()
    s = s.map(lambda t: unicodedata.normalize('NFD', t)
                                   .encode('ascii', 'ignore')
                                   .decode('utf-8'))
    s = s.str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()
    return s


# =============================================================================
# Exact matching
# =============================================================================

def _match_exact(main_keys, other_keys):
    """
    For each row of main, return the positional index of the first row of
    other with the same non-empty key, or -1. Empty strings never match.
    """
    lookup = {}
    for i, k in enumerate(other_keys):
        if k and k not in lookup:
            lookup[k] = i

    out = np.full(len(main_keys), -1, dtype=np.int64)
    for i, k in enumerate(main_keys):
        if k:
            j = lookup.get(k, -1)
            if j != -1:
                out[i] = j
    return out


# =============================================================================
# Blocking: word n-gram inverted index
# =============================================================================

def _word_ngrams(text: str, n: int) -> set[str]:
    """Return the set of word n-grams of a canonicalized title.
    Falls back to the set of single words when the title is shorter than n."""
    words = text.split()
    if len(words) < n:
        return set(words)
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _build_ngram_index(titles, n: int):
    """Build an inverted index: ngram -> list of positional indices of titles
    containing it."""
    index = defaultdict(list)
    for i, t in enumerate(titles):
        for ng in _word_ngrams(t, n):
            index[ng].append(i)
    return index


def _shortlist_candidates(title: str, index, n: int, max_candidates: int):
    """Return the set of positional indices in `other` that share at least
    one n-gram with `title`. Capped at `max_candidates` by overlap count."""
    ngrams = _word_ngrams(title, n)
    if not ngrams:
        return []

    # Count per candidate how many n-grams it shares with the query.
    scores = defaultdict(int)
    for ng in ngrams:
        for idx in index.get(ng, ()):
            scores[idx] += 1

    if len(scores) <= max_candidates:
        return list(scores.keys())

    # Too many candidates -> keep the ones with the most n-gram overlaps.
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_candidates]
    return [idx for idx, _ in top]


# =============================================================================
# Fuzzy matching with blocking
# =============================================================================

def _match_fuzzy_blocked(
    main_titles,
    other_titles,
    threshold,
    rows_to_check,
    ngram_size: int,
    max_candidates: int,
    scorer,
):
    """
    Fuzzy match using n-gram blocking. Returns an array of length
    len(main_titles) with the positional index of the best match in other
    (or -1 if none above threshold).

    Much faster than rapidfuzz.cdist on the full Cartesian product: each
    query only scores against its shortlist of candidates.
    """
    out = np.full(len(main_titles), -1, dtype=np.int64)
    if not rows_to_check.any() or len(other_titles) == 0:
        return out

    # Build the inverted index once, on the `other` side.
    index = _build_ngram_index(other_titles, n=ngram_size)

    subset_idx = np.where(rows_to_check)[0]

    for i in subset_idx:
        query = main_titles[i]
        candidates = _shortlist_candidates(query, index, ngram_size, max_candidates)
        if not candidates:
            continue

        # Score only against the shortlisted candidates.
        candidate_texts = [other_titles[c] for c in candidates]
        scores = rf_process.cdist(
            [query], candidate_texts, scorer=scorer,
        )[0]

        best_local = scores.argmax()
        if scores[best_local] >= threshold:
            out[i] = candidates[best_local]

    return out


# =============================================================================
# Public API
# =============================================================================

def merge_bibs(
    datasets,
    title_similarity_threshold,
    ngram_size: int,
    max_candidates_per_row: int,
    scorer,
):
    """
    Merge bibliographic DataFrames and drop duplicates.

    Priority rule, per row of main_ds:
      1. Normalized DOI equal to some row in other_ds     -> duplicate.
      2. Otherwise: canonical title equal                  -> duplicate.
      3. Otherwise: fuzzy title score >= threshold, using n-gram blocking
         to avoid the full Cartesian product.

    Parameters
    ----------
    datasets
    title_similarity_threshold
    ngram_size : int
        Word n-gram size for the blocking index. Larger = fewer but stricter
        candidates (3 is a reasonable choice for scientific titles).
    max_candidates_per_row : int
        Upper bound on the shortlist size per query. Prevents pathological
        cases where very common n-grams pull in thousands of candidates.
    scorer : callable
        rapidfuzz scorer used to compare shortlisted candidates
        (e.g. ``rapidfuzz.fuzz.token_set_ratio`` or ``fuzz.WRatio``).

    `doi_similarity_threshold` is unused (DOIs are compared exactly after
    normalization), kept in the signature for backward compatibility.
    """
    main_ds = datasets[0]

    for n in range(len(datasets) - 1):
        other_ds = datasets[n + 1]

        # --- Pass 1: DOI exact. ----------------------------------------------
        main_doi_norm  = _normalize_doi(main_ds['doi']).tolist()
        other_doi_norm = _normalize_doi(other_ds['doi']).tolist()
        doi_match = _match_exact(main_doi_norm, other_doi_norm)

        # --- Pass 2: canonical title exact. ----------------------------------
        main_title_canon  = _canonicalize_title(main_ds['title']).tolist()
        other_title_canon = _canonicalize_title(other_ds['title']).tolist()
        title_exact_match = _match_exact(main_title_canon, other_title_canon)
        title_exact_match[doi_match != -1] = -1

        # --- Pass 3: blocked fuzzy match on the rest. ------------------------
        not_matched_yet = (doi_match == -1) & (title_exact_match == -1)
        title_fuzzy_match = _match_fuzzy_blocked(
            main_title_canon, other_title_canon,
            title_similarity_threshold, not_matched_yet,
            ngram_size=ngram_size,
            max_candidates=max_candidates_per_row,
            scorer=scorer,
        )

        # --- Combine: DOI > exact title > fuzzy. -----------------------------
        drop_pos = np.where(doi_match != -1, doi_match,
                   np.where(title_exact_match != -1, title_exact_match,
                                                     title_fuzzy_match))
        drop_pos = np.unique(drop_pos[drop_pos >= 0])
        drop_labels = other_ds.index[drop_pos]

        other_ds = other_ds.drop(drop_labels)
        main_ds = pd.concat([main_ds, other_ds], ignore_index=True)

    return main_ds
