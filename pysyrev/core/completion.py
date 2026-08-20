"""
Metadata completion for bibliographic records.

No bibliographic source is complete: fields a record leaves empty can often be
recovered from another source that indexes the same work. This module gathers
those completion strategies. Each fills a specific set of columns by looking the
record up in an external provider (by DOI), while leaving every other field
untouched — completion never overwrites data already present. It is source-
agnostic: strategies run on the merged corpus, driven by the corpus-level
``bib.abstract_completion`` config (see :class:`AbstractCompletionConfig`), and
several providers can be chained so a later one only fills what earlier ones
missed. Abstract completion is the first strategy; others (e.g. citation counts,
references, alternative abstract sources) can be added following the same
contract.

Providers register a backend keyed by ``provider`` name in
:attr:`BibDataset._COMPLETERS`; ``wos`` is the only one implemented today.

Abstract completion
-------------------
A non-trivial share of records carry no abstract — OpenAlex, a common primary
source, simply has none for many works. Those records cannot be LLM-screened on
their abstract and, because :meth:`BibDataset.clean_and_drop` drops no-abstract
rows, they would be silently discarded, biasing the corpus.

:func:`complete_abstracts_from_wos` recovers those abstracts from Web of Science
Expanded by DOI lookup and fills **only** the ``abstract`` column. Every other
field — in particular ``references`` (the IDs used for coupling networks) — is
left untouched. WoS under-indexes preprints and many conference series, so this
is a completion pass, never a primary search.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from tqdm import tqdm

from pysyrev.core.api import WosClient
from pysyrev.core.bib import ABSTRACT, DOI
from pysyrev.core.mappers import from_wos_result


# Matches the bare DOI inside a URL-form or prefixed value, e.g.
# 'https://doi.org/10.1234/foo' -> '10.1234/foo'.
_DOI_RE = re.compile(r'10\.\d{4,9}/\S+')


def _is_missing(value) -> bool:
    """True when a cell holds nothing usable (None, NaN, empty/whitespace)."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _bare_doi(value) -> Optional[str]:
    """Extract the bare ``10.xxxx/...`` DOI from a raw cell.

    OpenAlex returns DOIs in URL form (``https://doi.org/10...``), which the
    WoS ``DO=(...)`` query does not understand. Returns None when no DOI-shaped
    token is present.
    """
    match = _DOI_RE.search(str(value).strip())
    return match.group(0) if match else None


def complete_abstracts_from_wos(dataframe: pd.DataFrame,
                                api_key: str,
                                session_file: Optional[str] = None,
                                show_progress: bool = True) -> int:
    """Fill missing abstracts in place from WoS Expanded, by DOI.

    For each row whose ``abstract`` is empty but which carries a DOI, query WoS
    (``DO=(<doi>)``) and, when a record with an abstract is found, write it back
    into the ``abstract`` column. No other column is modified — the OpenAlex
    ``references``/IDs used for coupling networks are preserved as-is.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DEFAULT_FIELDS dataframe (must have ``abstract`` and ``doi`` columns).
        Modified in place.
    api_key : str
        WoS Expanded API key.
    session_file : Union[None, str]
        Optional path for WoS pagination state. Mostly irrelevant for the
        single-record lookups done here.
    show_progress : bool
        Display a tqdm bar over the DOIs looked up.

    Returns
    -------
    int
        Number of abstracts recovered.
    """
    if ABSTRACT not in dataframe.columns or DOI not in dataframe.columns:
        raise ValueError(
            f"WoS abstract completion needs both {ABSTRACT!r} and {DOI!r} columns."
        )

    client = WosClient(api_key=api_key, session_file=session_file)

    # Rows worth a lookup: no abstract, but a usable DOI.
    todo = []
    for idx in dataframe.index:
        if not _is_missing(dataframe.at[idx, ABSTRACT]):
            continue
        doi = _bare_doi(dataframe.at[idx, DOI])
        if doi is not None:
            todo.append((idx, doi))

    recovered = 0
    for idx, doi in tqdm(todo, desc="WoS completion", disable=not show_progress):
        try:
            result = client.search(query=f"DO=({doi})",
                                    max_records=1,
                                    show_progress=False)
        except Exception:
            # A single failed lookup must not abort the whole completion pass.
            continue
        if not result.records:
            continue
        abstract = from_wos_result(result)[ABSTRACT].iloc[0]
        if not _is_missing(abstract):
            dataframe.at[idx, ABSTRACT] = abstract
            recovered += 1

    return recovered
