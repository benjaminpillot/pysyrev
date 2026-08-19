"""
Reference resolution: raw reference strings → internal document IDs.

Each document's ``references`` column contains either:
- OpenAlex work IDs (``https://openalex.org/Wxxxx; ...``) — produced by the
  OpenAlex mapper.
- Free-text WoS cited-reference strings (``Author, Year, Journal, V, P[, DOI]``)
  — produced by read_bib.

Resolution runs four passes in decreasing reliability order:

  1. Direct ID match — for OpenAlex references already stored as full IDs.
  2. Normalized DOI exact match — works across sources (WoS ref with DOI,
     OpenAlex document; or vice-versa) after stripping URL prefixes and
     lowercasing.
  2.5 Author + year lookup — targets WoS references without a DOI. Parses
     the first-author last name and year from the WoS reference string, then
     disambiguates multiple candidates by fuzzy journal name matching (low
     cutoff to account for WoS journal abbreviations).
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
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Author / year / journal helpers for WoS reference strings
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r'^\d{4}$')


def _normalize_lastname(name: str) -> str:
    norm = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^a-z]', '', norm.lower())


def _first_author_lastname(author_field) -> Optional[str]:
    """Extract and normalize the last name of the first author from a dataset author field.

    Handles 'Smith, John; Jones, Mary', 'Smith J; Jones M', 'Smith, John'.
    """
    if not author_field or pd.isna(author_field):
        return None
    first = str(author_field).split(';')[0].strip()
    lastname = re.split(r'[,\s]+', first)[0]
    return _normalize_lastname(lastname) if lastname else None


def _normalize_journal(journal: str) -> str:
    norm = unicodedata.normalize('NFD', journal.lower()).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^a-z0-9 ]', ' ', norm).strip()


def _parse_wos_ref(ref: str):
    """Parse a WoS cited-reference string into (lastname, year, journal).

    WoS format: 'Smith J, 2018, NAT COMMUN, V15, P123[, DOI ...]'
    Returns (None, None, None) if the format is not recognized.
    """
    parts = [p.strip() for p in ref.split(',')]
    if len(parts) < 2:
        return None, None, None

    author_tokens = parts[0].split()
    if not author_tokens:
        return None, None, None
    lastname = _normalize_lastname(author_tokens[0])

    if not _YEAR_RE.match(parts[1]):
        return None, None, None
    year = int(parts[1])

    journal = parts[2] if len(parts) > 2 else None
    return lastname, year, journal


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
    id_to_pos:          dict,
    doi_to_pos:         dict,
    author_year_to_pos: dict,
    canon_journals:     list,
    canon_titles:       list,
    ngram_index:        dict,
    ngram_size:         int,
    max_candidates:     int,
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

    # Pass 2.5: first-author last name + year, disambiguated by journal.
    # Targets WoS references that lack a DOI.
    lastname, year, ref_journal = _parse_wos_ref(ref)
    if lastname and year:
        candidates = author_year_to_pos.get((lastname, year), [])
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1 and ref_journal:
            # WoS uses abbreviated journal names, so use a low score cutoff.
            norm_ref_journal = _normalize_journal(ref_journal)
            texts = [canon_journals[c] for c in candidates]
            result = rf_process.extractOne(
                norm_ref_journal, texts,
                scorer=fuzz.token_set_ratio,
                score_cutoff=60,
            )
            if result is not None:
                return candidates[result[2]]

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

def flag_shared_unresolved_references(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add a 'shared_unresolved_references' column.

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    if 'unresolved_references' not in dataset.columns:
        raise ValueError(
            "No 'unresolved_references' column — run resolve_references() first."
        )

    sep = '; '
    parsed = dataset['unresolved_references'].apply(
        lambda x: x.split(sep) if isinstance(x, str) else []
    )

    counts = Counter(ref for refs in parsed for ref in refs)
    shared = {r for r, n in counts.items() if n >= 2}

    dataset['shared_unresolved_references'] = parsed.apply(
        lambda refs: sep.join(r for r in refs if r in shared) or np.nan
    )

    return dataset


def resolve_references(
    df: pd.DataFrame,
    cross_id_map:       Optional[dict] = None,
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
    cross_id_map : dict, optional
        Mapping ``{dropped_id: kept_id}`` produced by :func:`merge_bibs`.
        Allows references pointing to IDs that were dropped during deduplication
        (e.g. WoS IDs replaced by the OpenAlex duplicate, per the configured
        merge priority) to resolve to the surviving record.
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

    # Extend with alias IDs from cross-source deduplication (e.g. WoS IDs that
    # were dropped in favour of the OpenAlex duplicate during merge_bibs).
    for alias, canonical in (cross_id_map or {}).items():
        if alias not in id_to_pos and canonical in id_to_pos:
            id_to_pos[alias] = id_to_pos[canonical]

    doi_norm = _normalize_doi(df['doi'])
    doi_to_pos = {norm: i for i, norm in enumerate(doi_norm) if norm}

    author_year_to_pos: dict[tuple, list[int]] = {}
    for i, (author, yr) in enumerate(zip(df['author'], df['year'])):
        lastname = _first_author_lastname(author)
        if lastname and pd.notna(yr):
            key = (lastname, int(yr))
            author_year_to_pos.setdefault(key, []).append(i)

    canon_journals = [
        _normalize_journal(str(j)) if pd.notna(j) else ''
        for j in df['journal']
    ]

    canon_titles = _canonicalize_title(df['title']).tolist()
    ngram_index  = _build_ngram_index(canon_titles, n=ngram_size)
    doc_ids      = df['id'].tolist()

    # --- Resolve row by row --------------------------------------------------

    resolved_col   = []
    unresolved_col = []

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Resolving references"):
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
                author_year_to_pos, canon_journals,
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
