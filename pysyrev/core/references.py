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

Separately, :func:`complete_reference_keys` produces a ``reference_keys`` column:
each reference remapped to a **canonical DOI key** (native token kept when no DOI
is derivable). Unlike ``reference_ids`` — which only links references to other
corpus documents — this gives the extra-corpus works a shared identity too, so
bibliographic coupling and co-citation hold across merged sources regardless of
merge order. See the "Canonical reference keys" section below.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process
from tqdm import tqdm

from pysyrev.core.bib import DOI, ID, REFS
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


# ===========================================================================
# Canonical reference keys — a source-agnostic coupling basis
# ===========================================================================
#
# :func:`resolve_references` above links references to *corpus* documents (for
# the citation network). This section serves a different need: giving every
# reference — including the extra-corpus works that carry most of the coupling
# signal — a single shared identity so bibliographic coupling and co-citation
# hold across merged sources.
#
# The canonical key of a reference is its normalized **DOI** whenever one can be
# derived, else the reference's native token kept verbatim (so it still couples
# within its own id space). Because the DOI is the shared key, the coupling basis
# is independent of which source won the merge, and of which sources are used.
#
# The design is source-agnostic. Two of the three ways a token gets a DOI are
# generic; only the third — turning an *opaque, extra-corpus* id into a DOI —
# needs source-specific knowledge, isolated in the ``_ID_SCHEMES`` registry:
#   * the token is (or points at) a corpus document -> its DOI, looked up in the
#     ``id -> DOI`` table seeded for free from every corpus row's id + doi,
#     whatever the source (generic).
#   * the token carries a DOI (WoS / Scopus raw string, bare DOI, doi.org URL)
#     -> that normalized DOI (generic).
#   * the token is an opaque id of a *known scheme* (OpenAlex ``Wxxxx`` today;
#     Scopus EID / PMID could be added) pointing outside the corpus -> resolved
#     once to a DOI by that scheme's resolver and cached. No DOI on record, or no
#     resolver available -> the token is kept verbatim.
#   * anything else -> kept verbatim.

_OPENALEX_ID_RE = re.compile(r'(?:https?://openalex\.org/)?(W\d+)$', re.IGNORECASE)
_DOI_PREFIX_RE  = re.compile(r'^\s*(?:https?://)?(?:dx\.)?doi\.org/', re.IGNORECASE)
_BARE_DOI_RE    = re.compile(r'^10\.\d{4,9}/\S+$')

_CACHE_COLS = ('reference_id', 'doi')  # persistent id -> DOI cache schema
_ID_CHUNK   = 50                       # OpenAlex OR-filter cap for one `|` list


def _as_openalex_id(token) -> Optional[str]:
    """Return the bare OpenAlex work id (``W\\d+``) of *token*, or None."""
    if not isinstance(token, str):
        return None
    m = _OPENALEX_ID_RE.match(token.strip())
    return m.group(1).upper() if m else None


# Reference-id schemes: scheme name -> a normalizer that returns the token's
# canonical id form if it belongs to the scheme, else None. This is the only
# source-specific knowledge in this section; a matching resolver (built by the
# caller, e.g. from API credentials) turns those canonical ids into DOIs.
_ID_SCHEMES: Dict[str, Callable[[object], Optional[str]]] = {
    'openalex': _as_openalex_id,
}


def _normalize_id(token) -> Optional[str]:
    """Canonical id of *token* if it matches any known scheme, else None."""
    if not isinstance(token, str):
        return None
    for normalize in _ID_SCHEMES.values():
        nid = normalize(token)
        if nid is not None:
            return nid
    return None


def _id_key(token) -> Optional[str]:
    """Table key for *token*: its canonical scheme id, else the raw token.

    Falling back to the raw token means references expressed as a source's native
    id still match a corpus row keyed by that same id, even for a scheme we do not
    otherwise recognise.
    """
    if not isinstance(token, str):
        return None
    tok = token.strip()
    if not tok:
        return None
    return _normalize_id(tok) or tok


def _normalize_doi_value(doi) -> Optional[str]:
    """Canonicalize a single DOI: strip the doi.org prefix, lowercase, trim.
    Returns None for empty / non-DOI input."""
    if not isinstance(doi, str) or not doi.strip():
        return None
    d = _DOI_PREFIX_RE.sub('', doi.strip()).lower().rstrip('.,;')
    return d or None


def _canonical_key(token: str, table: Dict[str, Optional[str]]) -> str:
    """Map one reference token to its canonical key (see section header)."""
    tok = token.strip()

    hit = table.get(_id_key(tok))        # points at a known / resolved id?
    if hit:
        return hit

    doi = _extract_ref_doi(tok)          # DOI embedded in a raw citation string
    if doi:
        return _normalize_doi_value(doi) or tok

    if _BARE_DOI_RE.match(tok.lower()):  # already a bare DOI
        return _normalize_doi_value(tok) or tok

    return tok


# ---- id -> DOI table -------------------------------------------------------

def seed_table_from_dataframe(df: pd.DataFrame,
                              id_col: str = ID, doi_col: str = DOI) -> Dict[str, str]:
    """Build an ``{id: doi}`` table for free from the corpus rows.

    Every row that carries a DOI yields one entry, keyed by its id (whatever the
    source) — this covers all references pointing at documents inside the corpus,
    with no API call.
    """
    table: Dict[str, str] = {}
    if id_col not in df.columns or doi_col not in df.columns:
        return table
    for id_, doi in zip(df[id_col], df[doi_col]):
        key = _id_key(id_)
        norm = _normalize_doi_value(doi)
        if key and norm:
            table[key] = norm
    return table


def load_cache(path: Optional[str]) -> Dict[str, Optional[str]]:
    """Load the persistent ``id -> DOI`` CSV cache into a dict.

    A blank ``doi`` cell means "resolved, but the work has no DOI" (stored as
    None) so it is not re-queried. Missing / unreadable file -> empty table.
    """
    if not path:
        return {}
    try:
        cache = pd.read_csv(path, dtype=str)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        return {}
    table: Dict[str, Optional[str]] = {}
    for rid, doi in zip(cache.get('reference_id', []), cache.get('doi', [])):
        key = _id_key(rid)
        if key is not None:
            table[key] = doi if isinstance(doi, str) and doi.strip() else None
    return table


def save_cache(path: Optional[str], table: Dict[str, Optional[str]]) -> None:
    """Write the whole ``id -> DOI`` table to the CSV cache (None -> blank)."""
    if not path:
        return
    rows = [{'reference_id': rid, 'doi': doi or ''} for rid, doi in sorted(table.items())]
    pd.DataFrame(rows, columns=_CACHE_COLS).to_csv(path, index=False)


# ---- Extra-corpus resolution (source-specific resolvers) -------------------

def resolve_ids_via_openalex(ids: Iterable[str], client,
                             chunk: int = _ID_CHUNK) -> Dict[str, Optional[str]]:
    """OpenAlex resolver: map OpenAlex work ids to DOIs in batches of *chunk*.

    Returns ``{Wxxxx: doi_or_None}`` for every requested id. Ids the API does
    not return are mapped to None, so they are cached as "no DOI on record" and
    not queried again. This is the ``'openalex'`` entry of the scheme registry;
    the caller wires it up with a live :class:`OpenAlexClient`.
    """
    ids = [w for w in dict.fromkeys(_as_openalex_id(i) for i in ids) if w]
    out: Dict[str, Optional[str]] = {}
    for start in range(0, len(ids), chunk):
        batch = ids[start:start + chunk]
        filt = 'openalex_id:' + '|'.join(f'https://openalex.org/{w}' for w in batch)
        page = client._fetch_page({'filter': filt, 'select': 'id,doi', 'per-page': chunk})
        for rec in page.get('results', []):
            wid = _as_openalex_id(rec.get('id'))
            if wid is not None:
                out[wid] = _normalize_doi_value(rec.get('doi'))
        for w in batch:                  # ids not returned -> record as None
            out.setdefault(w, None)
    return out


# ---- Public API ------------------------------------------------------------

def _scheme_ref_ids(df: pd.DataFrame, normalize: Callable[[object], Optional[str]],
                    ref_col: str = REFS) -> Set[str]:
    """Distinct canonical ids of one scheme appearing in the references column."""
    ids: Set[str] = set()
    for val in df[ref_col]:
        if not isinstance(val, str):
            continue
        for tok in val.split(';'):
            nid = normalize(tok)
            if nid is not None:
                ids.add(nid)
    return ids


def _has_bridgeable_dois(df: pd.DataFrame, ref_col: str = REFS) -> bool:
    """True if any reference is a DOI-bearing token that is *not* a known id.

    Resolving extra-corpus ids to DOIs only bridges anything when another source
    contributes references already expressed as DOIs (WoS / Scopus strings, or
    bare DOIs). When the whole corpus references only same-scheme ids (e.g. a
    pure-OpenAlex run) there is nothing to bridge to, so the API step is pure cost
    with no coupling gain and is skipped.
    """
    for val in df[ref_col]:
        if not isinstance(val, str):
            continue
        for tok in val.split(';'):
            tok = tok.strip()
            if not tok or _normalize_id(tok) is not None:
                continue
            if _extract_ref_doi(tok) or _BARE_DOI_RE.match(tok.lower()):
                return True
    return False


def has_unkeyed_references(df: pd.DataFrame, ref_col: str = REFS) -> bool:
    """True if any reference is a raw token that is *not* an internal id key.

    A reference "points to a key" when it is a recognized scheme id (an OpenAlex
    ``Wxxxx`` id, etc.) that already names a document by its native id. Raw
    citation strings and bare DOIs (WoS / Scopus style) do *not*, so they need
    :func:`resolve_references` to be matched against the corpus. When every
    reference is already a scheme id (e.g. a single-source OpenAlex run), or there
    are no references at all (an API source that returns none), there is nothing to
    resolve and the caller can skip the pass.
    """
    if ref_col not in df.columns:
        return False
    for val in df[ref_col]:
        if not isinstance(val, str):
            continue
        for tok in val.split(';'):
            tok = tok.strip()
            if tok and _normalize_id(tok) is None:
                return True
    return False


def add_reference_keys(df: pd.DataFrame, table: Dict[str, Optional[str]],
                       ref_col: str = REFS,
                       out_col: str = 'reference_keys') -> pd.DataFrame:
    """Add *out_col*: each doc's references remapped to canonical keys."""
    df = df.copy()
    keyed: List[Optional[str]] = []
    for val in df[ref_col]:
        if not isinstance(val, str) or not val.strip():
            keyed.append(None)
            continue
        keys = [_canonical_key(tok, table) for tok in val.split(';') if tok.strip()]
        keyed.append('; '.join(keys) if keys else None)
    df[out_col] = keyed
    return df


def complete_reference_keys(df: pd.DataFrame,
                            cache_path: Optional[str] = None,
                            resolvers: Optional[Dict[str, Callable]] = None,
                            resolve_external: bool = True,
                            ref_col: str = REFS,
                            out_col: str = 'reference_keys') -> pd.DataFrame:
    """Compute the canonical ``reference_keys`` column for *df*.

    1. Seed the ``id -> DOI`` table for free from the corpus rows (any source)
       and from the persistent CSV cache.
    2. If *resolve_external*, for each id scheme with a supplied resolver, resolve
       the extra-corpus ids of that scheme still missing a DOI, and cache them.
    3. Remap every reference to its canonical key into *out_col*.

    *resolvers* maps an ``_ID_SCHEMES`` name (e.g. ``'openalex'``) to a callable
    ``ids -> {id: doi_or_None}`` — the caller builds these from API credentials
    (see :func:`resolve_ids_via_openalex`). When empty (or *resolve_external* is
    False) only the free, offline mapping is applied: extra-corpus ids without a
    cached DOI keep their native token.
    """
    resolvers = resolvers or {}
    table: Dict[str, Optional[str]] = load_cache(cache_path)
    table.update(seed_table_from_dataframe(df))   # corpus rows override the cache

    # Only pay for API resolution when another source contributes DOI-bearing
    # references to bridge to (see _has_bridgeable_dois).
    if resolve_external and resolvers and _has_bridgeable_dois(df, ref_col):
        changed = False
        for name, resolve in resolvers.items():
            normalize = _ID_SCHEMES.get(name)
            if normalize is None:
                continue
            missing = _scheme_ref_ids(df, normalize, ref_col=ref_col) - table.keys()
            if missing:
                table.update(resolve(missing))
                changed = True
        if changed:
            save_cache(cache_path, table)

    return add_reference_keys(df, table, ref_col=ref_col, out_col=out_col)
