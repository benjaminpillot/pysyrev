"""
Seed-driven corpus expansion — provider-agnostic core.

Turns a handful of *seed papers* (DOIs, or a provider's own work ids) into a
pool of candidate records: the recall-oriented alternative to a single query
when a subfield has no reliable keyword handle.

The pool is the union of four arms:

  1. the **seeds** themselves,
  2. **forward citations** — works citing a seed,
  3. **backward citations** — the works a seed cites,
  4. **text queries** — title/abstract searches in the field's vocabulary,
     the recall backstop for works that neither cite nor are cited by a seed.

Everything in this module is source-independent: it orchestrates the arms,
unions their results, and deduplicates the pool. The arms themselves — how one
asks a given API who cites what — live in a :class:`SeedExpander` subclass, one
per provider (:mod:`pysyrev.core.seed_expansion.openalex` today). Adding a
provider means implementing that class; nothing here changes.

The output is a **candidate pool, not a corpus**: expansion deliberately
over-collects. Filtering by year, document type and language is left to the
existing ``bib.extract`` step, and scope screening to the ``review`` stage.

Usage
-----
  from pysyrev.core.seed_expansion import OpenAlexExpander, read_seed_file

  expander = OpenAlexExpander(client)
  result = expander.expand(read_seed_file('seeds.txt'),
                           queries=['agent-based energy transition'],
                           year_min=2010)
  result.dataset   # DEFAULT_FIELDS DataFrame — the candidate pool
  result.counts    # per-arm sizes: seeds, forward, backward, queries, …
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from pysyrev.core.bib import ABSTRACT, DOI, ID, REFS, TITLE

_DOI_URL_PREFIX = re.compile(r'^https?://(dx\.)?doi\.org/', re.IGNORECASE)
_NON_ALNUM      = re.compile(r'[^a-z0-9]')

# Records for the same work rarely repeat verbatim across providers (and even
# within one), so titles are compared on a normalized key truncated to a length
# that is discriminative without being brittle.
_TITLE_KEY_LENGTH = 90


# =============================================================================
# Seed input (provider-independent)
# =============================================================================

def normalize_doi(value) -> Optional[str]:
    """Return the bare lowercase DOI of *value*, or None if it is not a DOI.

    Accepts a bare DOI or a ``doi.org`` URL. DOIs are the one identifier every
    provider understands, so this is the shared half of any expander's
    :meth:`SeedExpander.normalize_seed`.
    """
    if not isinstance(value, str):
        return None
    token = _DOI_URL_PREFIX.sub('', value.strip())
    return token.lower() if token.lower().startswith('10.') else None


def read_seed_file(path: str) -> List[str]:
    """Read a seed file — one DOI (or provider work id) per line.

    Blank lines and lines starting with ``#`` are ignored; on a line carrying
    extra text (e.g. ``10.1000/aaa   Smith et al. 2020``) only the first
    whitespace-separated token is read, since no identifier contains
    whitespace. Tokens are returned as written — validating them is the
    expander's job (:meth:`SeedExpander.normalize_seed`).
    """
    seeds: List[str] = []
    with open(path, 'r', encoding='utf-8-sig') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            seeds.append(line.split()[0])
    return seeds


# =============================================================================
# Deduplication (provider-independent, on the DEFAULT_FIELDS schema)
# =============================================================================

def title_key(title) -> Optional[str]:
    """Normalized title key used to catch a provider's duplicate records."""
    if not isinstance(title, str):
        return None
    return _NON_ALNUM.sub('', title.lower())[:_TITLE_KEY_LENGTH] or None


def _text_length(value) -> int:
    """Length of a text field, 0 when it is missing."""
    return len(value) if isinstance(value, str) else 0


def dedup_pool(dataset: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate a pool by work id **and** normalized title key.

    Providers emit duplicate records for the same work, sometimes with an empty
    DOI or a missing abstract on one copy, so an id-only dedup leaves doubles;
    the title key catches them and the richest copy is kept — the one carrying
    a DOI, then the longer abstract, then the longer reference list. Untitled
    records are compared on their id alone.

    This runs *within* one source, before the cross-source ``bib.merge`` — an
    expansion pool is assembled from several API calls, so it can carry
    duplicates that a single query never would.
    """
    if dataset.empty:
        return dataset

    dataset = dataset.drop_duplicates(subset=[ID], keep='first')

    keys = dataset[TITLE].map(title_key)
    ranked = dataset.assign(
        _key      = keys.where(keys.notna(), dataset[ID]),
        _has_doi  = dataset[DOI].map(lambda v: bool(_text_length(v))),
        _abstract = dataset[ABSTRACT].map(_text_length),
        _refs     = dataset[REFS].map(_text_length),
    ).sort_values(['_has_doi', '_abstract', '_refs'],
                  ascending=False, kind='stable')
    kept = ranked.drop_duplicates(subset='_key', keep='first').index

    # Keep the original record order — dedup should not reshuffle the pool.
    return dataset[dataset.index.isin(kept)].reset_index(drop=True)


# =============================================================================
# Expansion result
# =============================================================================

@dataclass
class SeedExpansionResult:
    """Outcome of one :meth:`SeedExpander.expand` run.

    Attributes
    ----------
    dataset : pd.DataFrame
        The candidate pool, deduplicated, in the DEFAULT_FIELDS schema.
    seed_ids : list of str
        Provider ids of the seeds that resolved — a subset of the pool's ids.
    unresolved_seeds : list of str
        Seed tokens the provider knows nothing about, plus tokens that were
        not a valid identifier for it.
    counts : dict
        Per-arm sizes (``seeds``, ``forward``, ``backward``, ``queries``)
        followed by ``raw_union`` and ``deduped``.
    """
    dataset:          pd.DataFrame   = field(default_factory=pd.DataFrame)
    seed_ids:         List[str]      = field(default_factory=list)
    unresolved_seeds: List[str]      = field(default_factory=list)
    counts:           Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable recap of the expansion."""
        arms = ', '.join(f'{name}: {n}' for name, n in self.counts.items())
        line = f"Seed expansion — {arms}"
        if self.unresolved_seeds:
            line += f" (unresolved seeds: {', '.join(self.unresolved_seeds)})"
        return line


# =============================================================================
# Provider interface
# =============================================================================

class SeedExpander(ABC):
    """Provider-specific arms of a seed expansion.

    A subclass wires the four arms to one API and maps its records onto the
    DEFAULT_FIELDS schema. Records stay opaque provider payloads throughout —
    each arm returns them keyed by that provider's work id, and only
    :meth:`to_dataframe` interprets them — so this class is the *only* place a
    new source has to touch.

    An arm a provider cannot serve (no citation graph, no full-text search) may
    be declared unsupported through the ``supports_*`` flags; the orchestration
    then skips it and records a zero count instead of failing.

    Subclasses implement:
      * :meth:`normalize_seed` — accepted spellings of a seed identifier,
      * :meth:`fetch_seeds`, :meth:`forward_citations`,
        :meth:`backward_citations`, :meth:`search_queries` — the arms,
      * :meth:`to_dataframe` — records → DEFAULT_FIELDS.
    """

    #: Provider name, as used in the config and in log lines.
    name: str = 'expander'

    #: Arms this provider can serve. An unsupported arm is skipped silently.
    supports_forward:  bool = True
    supports_backward: bool = True
    supports_queries:  bool = True

    # ---- provider hooks ----------------------------------------------------

    @abstractmethod
    def normalize_seed(self, token: str) -> Optional[str]:
        """Normalize one seed token to an identifier this provider accepts,
        or return None when the token is not one (it is then reported as
        unresolved). Every provider should accept a DOI — see
        :func:`normalize_doi`."""

    @abstractmethod
    def fetch_seeds(self, seeds: Sequence[str]) -> Tuple[Dict[str, dict], List[str]]:
        """Fetch the seed records themselves.

        Returns ``({work_id: record}, unresolved)`` where *unresolved* lists the
        normalized seed tokens the provider returned nothing for."""

    @abstractmethod
    def forward_citations(self, seed_records: Dict[str, dict],
                          year_min: Optional[int] = None,
                          year_max: Optional[int] = None) -> Dict[str, dict]:
        """Works **citing** the seeds, unioned across seeds. The year window is
        a retrieval bound — apply it server-side when the provider supports it,
        ignore it otherwise (``bib.extract`` filters for real)."""

    @abstractmethod
    def backward_citations(self, seed_records: Dict[str, dict]) -> Dict[str, dict]:
        """Works the seeds **cite** — their reference lists, fetched in full."""

    @abstractmethod
    def search_queries(self, queries: Sequence[str],
                       year_min: Optional[int] = None,
                       year_max: Optional[int] = None) -> Dict[str, dict]:
        """Union of title/abstract searches over *queries*.

        Each entry is one query in the provider's own search syntax — how its
        terms combine (AND-ed, phrase-matched, stemmed) is the provider's
        business, so document it on the override."""

    @abstractmethod
    def to_dataframe(self, records: Iterable[dict]) -> pd.DataFrame:
        """Map provider records onto the DEFAULT_FIELDS schema."""

    # ---- orchestration (shared by every provider) --------------------------

    def expand(self,
               seeds: Iterable[str],
               queries: Optional[Sequence[str]] = None,
               year_min: Optional[int] = None,
               year_max: Optional[int] = None,
               use_forward: bool = True,
               use_backward: bool = True) -> SeedExpansionResult:
        """Expand *seeds* into a deduplicated candidate pool.

        Parameters
        ----------
        seeds : iterable of str
            Seed DOIs and/or provider work ids. Typically 5-15 papers
            unambiguously in the subfield, spanning its sub-themes so the
            citation arms reach every corner of it.
        queries : sequence of str, optional
            Title/abstract searches in the field's own vocabulary (10-25 is a
            good range). Each one is a separate query and the results are
            unioned; how a single query is parsed — AND-ed terms, phrase
            match, stemming — is the provider's business (see
            :meth:`search_queries`).
        year_min, year_max : int, optional
            Publication-year window used as a *retrieval* bound on the arms
            that support it. Seeds outside the window still drive the
            expansion; definitive filtering belongs to ``bib.extract``.
        use_forward, use_backward : bool
            Toggle the citation arms. Disabling both leaves a query-only pool
            — a plain keyword corpus, useful as a comparison baseline.

        Returns
        -------
        SeedExpansionResult
        """
        tokens, unresolved = [], []
        for token in seeds:
            seed_id = self.normalize_seed(token)
            (tokens if seed_id else unresolved).append(seed_id or token)
        if not tokens and not unresolved:
            raise ValueError(
                "Seed expansion has no seed to start from — the seed file is "
                "empty (or holds only comments) and no inline seed is declared.")
        if not tokens:
            raise ValueError(
                f"No usable seed for the {self.name} expander: none of "
                f"{unresolved!r} is a DOI or a valid {self.name} work id.")

        seed_records, missing = self.fetch_seeds(tokens)
        unresolved += list(missing)

        union = dict(seed_records)          # seeds are always candidates
        counts = {'seeds': len(seed_records)}

        if use_forward and self.supports_forward:
            arm = self.forward_citations(seed_records, year_min, year_max)
            counts['forward'] = len(arm)
            union.update(arm)

        if use_backward and self.supports_backward:
            arm = self.backward_citations(seed_records)
            counts['backward'] = len(arm)
            union.update(arm)

        if queries and self.supports_queries:
            arm = self.search_queries(queries, year_min, year_max)
            counts['queries'] = len(arm)
            union.update(arm)

        dataset = dedup_pool(self.to_dataframe(union.values()))
        counts.update({'raw_union': len(union), 'deduped': len(dataset)})

        return SeedExpansionResult(
            dataset          = dataset,
            seed_ids         = list(seed_records),
            unresolved_seeds = unresolved,
            counts           = counts,
        )
