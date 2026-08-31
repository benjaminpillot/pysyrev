"""
API response → DEFAULT_FIELDS mappers.

Each public function converts the raw JSON returned by an API client into a
pandas DataFrame that conforms to the DEFAULT_FIELDS schema defined in bib.py.
One function per source; private helpers are prefixed with the source name.

Adding a new source: write a _map_<source>_record() function plus the
corresponding from_<source>_result() entry point, following the pattern below.
"""

from typing import Optional

import pandas as pd

from pysyrev.core.bib import DEFAULT_FIELDS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_dataframe(rows: list) -> pd.DataFrame:
    """Build a DEFAULT_FIELDS DataFrame from a list of record dicts and
    cast numeric columns to a numeric dtype."""
    df = pd.DataFrame(rows, columns=list(DEFAULT_FIELDS.keys()))
    for col, kind in DEFAULT_FIELDS.items():
        if kind == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# OpenAlex API → DEFAULT_FIELDS
# ---------------------------------------------------------------------------

def _oa_reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct plain text from an OpenAlex abstract_inverted_index.

    OpenAlex stores abstracts as {word: [position, ...]} dicts to avoid
    copyright issues. This reverses the index back into a space-joined string.
    """
    if not inverted_index:
        return None
    pos_to_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_to_word[pos] = word
    if not pos_to_word:
        return None
    return ' '.join(pos_to_word[i] for i in range(max(pos_to_word) + 1)
                    if i in pos_to_word)


def _oa_authorships(record: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract '; '-joined author display names and raw affiliation strings from
    an OpenAlex record's ``authorships`` list (matching the file path's
    ``authorships.author.display_name`` / ``authorships.raw_affiliation_strings``
    mapping and the dataset's '; ' author convention)."""
    authorships = record.get('authorships') or []
    authors = '; '.join(
        a['author']['display_name']
        for a in authorships
        if a.get('author') and a['author'].get('display_name')
    ) or None
    affiliations = '; '.join(
        aff
        for a in authorships
        for aff in (a.get('raw_affiliation_strings') or [])
        if aff
    ) or None
    return authors, affiliations


def _map_openalex_record(record: dict) -> dict:
    keywords = record.get('keywords') or []
    keyword_str = '; '.join(
        kw.get('display_name', '') for kw in keywords if kw.get('display_name')
    ) or None

    references = record.get('referenced_works') or []
    references_str = '; '.join(references) if references else None

    source = ((record.get('primary_location') or {}).get('source') or {})

    author, affiliations = _oa_authorships(record)

    return {
        'id':              record.get('id'),
        'author':          author,
        'affiliations':    affiliations,
        'abstract':        _oa_reconstruct_abstract(record.get('abstract_inverted_index')),
        'journal':         source.get('display_name'),
        'title':           record.get('title'),
        'author_keywords': keyword_str,
        'doi':             record.get('doi'),
        'language':        record.get('language'),
        'document_type':   record.get('type'),
        'references':      references_str,
        'year':            record.get('publication_year'),
        'cited_by':        record.get('cited_by_count'),
    }


def from_openalex_records(records) -> pd.DataFrame:
    """Convert an iterable of raw OpenAlex works to a DEFAULT_FIELDS DataFrame.

    Use this when the records come from somewhere other than a single search —
    a seed expansion unions several queries (see
    :mod:`pysyrev.core.seed_expansion`).
    """
    return _to_dataframe([_map_openalex_record(r) for r in records])


def from_openalex_result(result) -> pd.DataFrame:
    """Convert an :class:`OpenAlexSearchResult` to a DEFAULT_FIELDS DataFrame.

    Parameters
    ----------
    result : OpenAlexSearchResult
        Raw result returned by :meth:`OpenAlexClient.search`.
    """
    return from_openalex_records(result.records)


# ---------------------------------------------------------------------------
# WoS Expanded API → DEFAULT_FIELDS
# ---------------------------------------------------------------------------

def _wos_as_list(val) -> list:
    """Normalize a WoS field that the API returns as either a single item or
    a list depending on the record count."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _wos_title(record: dict) -> Optional[str]:
    titles = _wos_as_list(
        record.get('static_data', {}).get('summary', {})
              .get('titles', {}).get('title')
    )
    for t in titles:
        if isinstance(t, dict) and t.get('type') == 'item':
            return t.get('content')
    return None


def _wos_journal(record: dict) -> Optional[str]:
    titles = _wos_as_list(
        record.get('static_data', {}).get('summary', {})
              .get('titles', {}).get('title')
    )
    for t in titles:
        if isinstance(t, dict) and t.get('type') == 'source':
            return t.get('content')
    return None


def _wos_abstract(record: dict) -> Optional[str]:
    abstract_node = (
        record.get('static_data', {}).get('fullrecord_metadata', {})
              .get('abstracts', {}).get('abstract')
    )
    if not abstract_node:
        return None
    abstract_node = _wos_as_list(abstract_node)[0]
    p = abstract_node.get('abstract_text', {}).get('p')
    if isinstance(p, list):
        return ' '.join(str(x) for x in p if x)
    return str(p) if p else None


def _wos_language(record: dict) -> Optional[str]:
    langs = _wos_as_list(
        record.get('static_data', {}).get('summary', {})
              .get('languages', {}).get('language')
    )
    for lang in langs:
        if isinstance(lang, dict) and lang.get('type') == 'primary':
            return lang.get('content')
    if langs:
        first = langs[0]
        return first.get('content') if isinstance(first, dict) else str(first)
    return None


def _wos_doctype(record: dict) -> Optional[str]:
    dt = (
        record.get('static_data', {}).get('summary', {})
              .get('doctypes', {}).get('doctype')
    )
    if dt is None:
        return None
    return _wos_as_list(dt)[0] if isinstance(dt, list) else str(dt)


def _wos_keywords(record: dict) -> Optional[str]:
    kws = (
        record.get('static_data', {}).get('fullrecord_metadata', {})
              .get('keywords', {}).get('keyword')
    )
    if kws is None:
        return None
    kw_list = [str(k) for k in _wos_as_list(kws) if k]
    return '; '.join(kw_list) or None


def _wos_doi(record: dict) -> Optional[str]:
    identifiers = _wos_as_list(
        record.get('dynamic_data', {}).get('cluster_related', {})
              .get('identifiers', {}).get('identifier')
    )
    for ident in identifiers:
        if isinstance(ident, dict) and ident.get('type') == 'doi':
            return ident.get('value')
    return None


def _wos_cited_by(record: dict) -> Optional[int]:
    # silo_tc is always a list in the Expanded API — one entry per indexed
    # database (WOS, WOK, INSPEC, …). Use the WOS Core Collection count;
    # fall back to WOK (all-databases total) if WOS is absent.
    try:
        silo_tc = record['dynamic_data']['citation_related']['tc_list']['silo_tc']
        if isinstance(silo_tc, dict):
            return int(silo_tc['local_count'])
        for coll_id in ('WOS', 'WOK'):
            match = next((x for x in silo_tc if x.get('coll_id') == coll_id), None)
            if match is not None:
                return int(match['local_count'])
        return None
    except (KeyError, TypeError, ValueError):
        return None


def _wos_references(record: dict) -> Optional[str]:
    # The WoS Expanded search endpoint does NOT include references in-line.
    # Fetching them requires a separate call to GET /api/wos/{uid}/references.
    # Until that is implemented, this field is always None for API-sourced records.
    refs = _wos_as_list(
        record.get('static_data', {}).get('fullrecord_metadata', {})
              .get('references', {}).get('reference')
    )
    uids = [r['uid'] for r in refs if isinstance(r, dict) and r.get('uid')]
    return '; '.join(uids) if uids else None


def _map_wos_record(record: dict) -> dict:
    return {
        'id':              record.get('UID'),
        'title':           _wos_title(record),
        'journal':         _wos_journal(record),
        'abstract':        _wos_abstract(record),
        'year':            record.get('static_data', {}).get('summary', {})
                                 .get('pub_info', {}).get('pubyear'),
        'document_type':   _wos_doctype(record),
        'language':        _wos_language(record),
        'author_keywords': _wos_keywords(record),
        'doi':             _wos_doi(record),
        'cited_by':        _wos_cited_by(record),
        'references':      _wos_references(record),
    }


def from_wos_result(result) -> pd.DataFrame:
    """Convert a :class:`WosSearchResult` to a DEFAULT_FIELDS DataFrame.

    Parameters
    ----------
    result : WosSearchResult
        Raw result returned by :meth:`WosClient.search`.
    """
    return _to_dataframe([_map_wos_record(r) for r in result.records])
