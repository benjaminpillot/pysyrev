"""
Reference resolution: raw reference strings → internal document IDs.

Each document's ``references`` column contains either:
- OpenAlex work IDs (``https://openalex.org/Wxxxx; ...``) — produced by the
  OpenAlex mapper.
- Free-text WoS cited-reference strings (``Author, Year, Journal, V, P[, DOI]``)
  — produced by read_bib.

Resolution runs three passes in decreasing reliability order:

  1. Direct ID match — for OpenAlex references already stored as full IDs.
  2. Normalized DOI exact match — works across sources (WoS ref with DOI,
     OpenAlex document; or vice-versa) after stripping URL prefixes and
     lowercasing.
  3. Fuzzy title fallback — n-gram blocking + rapidfuzz, controlled by
     ``fuzzy_score_cutoff``. Useful when the raw reference happens to contain
     a recognizable title fragment; set cutoff to 100 to disable entirely.

Output columns (added to a copy of the input DataFrame):
  ``reference_ids``         — '; '-joined internal doc IDs of resolved refs.
  ``unresolved_references`` — '; '-joined raw strings that found no match.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from pysyrev.core.merge_bibs import (
    _normalize_doi,
    _canonicalize_title,
    _build_ngram_index,
    _shortlist_candidates,
)


# ---------------------------------------------------------------------------
# DOI extraction from raw reference strings
# ---------------------------------------------------------------------------

_DOI_LABEL_RE = re.compile(r'(?:^|[\s,;])DOI\s+(\S+)', re.IGNORECASE)
_DOI_URL_RE   = re.compile(r'https?://(?:dx\.)?doi\.org/(\S+)', re.IGNORECASE)


def _extract_ref_doi(ref: str) -> Optional[str]:
    """Return the normalized bare DOI found in *ref*, or None.

    Handles:
    - URL form  : ``https://doi.org/10.xxx/yyy``
    - WoS label : ``..., DOI 10.xxx/yyy``
    """
    m = _DOI_URL_RE.search(ref)
    if m:
        return m.group(1).lower().rstrip('.,;')
    m = _DOI_LABEL_RE.search(ref)
    if m:
        return m.group(1).lower().rstrip('.,;')
    return None


# ---------------------------------------------------------------------------
# Single-reference resolver
# ---------------------------------------------------------------------------

def _resolve_one(
    ref: str,
    id_to_pos:    dict,
    doi_to_pos:   dict,
    canon_titles: list,
    ngram_index:  dict,
    ngram_size:       int,
    max_candidates:   int,
    fuzzy_score_cutoff: int,
    scorer,
) -> Optional[int]:
    """Return the 0-based positional index of the matching document, or None."""

    # Pass 1: direct ID lookup (OpenAlex IDs already stored verbatim).
    pos = id_to_pos.get(ref)
    if pos is not None:
        return pos

    # Pass 2: DOI extraction + normalized exact match.
    doi = _extract_ref_doi(ref)
    if doi:
        pos = doi_to_pos.get(doi)
        if pos is not None:
            return pos

    # Pass 3: fuzzy title fallback (best-effort).
    if fuzzy_score_cutoff < 100:
        query = ref.lower()
        query = (unicodedata.normalize('NFD', query)
                             .encode('ascii', 'ignore')
                             .decode('utf-8'))
        query = re.sub(r'[^a-z0-9]+', ' ', query).strip()

        candidates = _shortlist_candidates(query, ngram_index, ngram_size, max_candidates)
        if candidates:
            texts = [canon_titles[c] for c in candidates]
            result = rf_process.extractOne(
                query, texts,
                scorer=scorer,
                score_cutoff=fuzzy_score_cutoff,
            )
            if result is not None:
                return candidates[result[2]]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_references(
    df: pd.DataFrame,
    fuzzy_score_cutoff: int = 90,
    ngram_size:         int = 3,
    max_candidates:     int = 50,
    scorer=fuzz.token_set_ratio,
) -> pd.DataFrame:
    """Resolve the raw ``references`` column to internal document IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset conforming to ``DEFAULT_FIELDS``.
    fuzzy_score_cutoff : int
        Minimum rapidfuzz score (0-100) to accept a fuzzy title match.
        Pass 100 to disable fuzzy matching entirely.
    ngram_size : int
        Word n-gram size for the blocking index used in the fuzzy pass.
    max_candidates : int
        Maximum candidates per query in the blocking phase.
    scorer : callable
        rapidfuzz scorer for the fuzzy title comparison.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two additional columns:

        ``reference_ids``
            Internal doc IDs of resolved references ('; '-joined), or None.
        ``unresolved_references``
            Raw reference strings that could not be matched ('; '-joined),
            or None.
    """
    df = df.copy()

    # --- Build lookup indices ------------------------------------------------

    id_to_pos = {id_: i for i, id_ in enumerate(df['id']) if pd.notna(id_)}

    doi_norm = _normalize_doi(df['doi'])
    doi_to_pos = {norm: i for i, norm in enumerate(doi_norm) if norm}

    canon_titles = _canonicalize_title(df['title']).tolist()
    ngram_index  = _build_ngram_index(canon_titles, n=ngram_size)
    doc_ids      = df['id'].tolist()

    # --- Resolve row by row --------------------------------------------------

    resolved_col   = []
    unresolved_col = []

    for _, row in df.iterrows():
        raw = row['references']
        if pd.isna(raw) or not str(raw).strip():
            resolved_col.append(None)
            unresolved_col.append(None)
            continue

        refs = [r.strip() for r in str(raw).split(';') if r.strip()]
        resolved   = []
        unresolved = []

        for ref in refs:
            pos = _resolve_one(
                ref,
                id_to_pos, doi_to_pos,
                canon_titles, ngram_index,
                ngram_size, max_candidates,
                fuzzy_score_cutoff, scorer,
            )
            if pos is not None:
                resolved.append(str(doc_ids[pos]))
            else:
                unresolved.append(ref)

        resolved_col.append('; '.join(resolved) if resolved else None)
        unresolved_col.append('; '.join(unresolved) if unresolved else None)

    df['reference_ids']         = resolved_col
    df['unresolved_references'] = unresolved_col
    return df
