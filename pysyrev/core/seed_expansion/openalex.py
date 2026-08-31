"""
OpenAlex backend for seed-driven corpus expansion.

Wires the four expansion arms (see :mod:`pysyrev.core.seed_expansion.base`) to
the OpenAlex /works endpoint:

  * seeds and backward citations — batched ``openalex_id:`` / ``doi:`` filters,
  * forward citations — ``cites:<id>``,
  * text queries — ``title_and_abstract.search:<query>``.

Records are raw OpenAlex work JSON throughout; :meth:`OpenAlexExpander.to_dataframe`
hands them to the existing OpenAlex mapper.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from pysyrev.core.mappers import from_openalex_records
from pysyrev.core.seed_expansion.base import SeedExpander, normalize_doi

_OA_URL_PREFIX = 'https://openalex.org/'
_OA_ID_PATTERN = re.compile(r'^W\d+$', re.IGNORECASE)

# Ids per batched /works request. OpenAlex ORs up to 100 values in one filter,
# but DOIs are long — 50 keeps the URL comfortably short.
_ID_CHUNK = 50


class OpenAlexExpander(SeedExpander):
    """Seed expansion over the OpenAlex API.

    Parameters
    ----------
    client : OpenAlexClient
        Live client; it carries the credentials, the polite-pool email, the
        retry policy and the optional resume cache.
    max_per_seed : int
        Cap on the forward citations pulled per seed. A highly-cited seed can
        otherwise return tens of thousands of works.
    max_per_query : int
        Cap on the works pulled per text query.
    show_progress : bool
        Display a progress bar over each arm.
    """

    name = 'openalex'

    def __init__(self, client, max_per_seed: int = 600, max_per_query: int = 400,
                 show_progress: bool = True):
        self.client = client
        self.max_per_seed = max_per_seed
        self.max_per_query = max_per_query
        self.show_progress = show_progress

    # ---- identifiers -------------------------------------------------------

    def normalize_seed(self, token: str) -> Optional[str]:
        """Accept a DOI (bare or ``doi.org`` URL) or an OpenAlex work id (bare
        or ``openalex.org`` URL). DOIs are returned as ``doi:10.xxxx/yyy`` —
        the form both take in an OpenAlex filter."""
        if not isinstance(token, str):
            return None
        candidate = token.strip()
        if candidate.lower().startswith(_OA_URL_PREFIX):
            candidate = candidate.rsplit('/', 1)[-1]
        if _OA_ID_PATTERN.match(candidate):
            return candidate.upper()
        doi = normalize_doi(token)
        return f'doi:{doi}' if doi else None

    @staticmethod
    def _work_id(record: dict) -> Optional[str]:
        """Bare ``Wxxxx`` id of a work record (OpenAlex ids are URLs)."""
        work_id = record.get('id')
        if not isinstance(work_id, str):
            return None
        return work_id.rsplit('/', 1)[-1] or None

    # ---- arms --------------------------------------------------------------

    def fetch_seeds(self, seeds: Sequence[str]) -> Tuple[Dict[str, dict], List[str]]:
        records = self._fetch_by_ids(seeds, desc='Seeds')
        resolved = set()
        for record in records.values():
            resolved.add(self._work_id(record))
            doi = normalize_doi(record.get('doi'))
            if doi:
                resolved.add(f'doi:{doi}')
        return records, [s for s in seeds if s not in resolved]

    def forward_citations(self, seed_records: Dict[str, dict],
                          year_min: Optional[int] = None,
                          year_max: Optional[int] = None) -> Dict[str, dict]:
        work_ids = [wid for wid in (self._work_id(r) for r in seed_records.values()) if wid]
        found: Dict[str, dict] = {}
        for work_id in tqdm(work_ids, desc='Forward citations',
                            disable=not self.show_progress or not work_ids):
            filters = {'cites': work_id}
            filters.update(self._year_filters(year_min, year_max))
            found.update(self._search(filters, self.max_per_seed))
        return found

    def backward_citations(self, seed_records: Dict[str, dict]) -> Dict[str, dict]:
        references: List[str] = []
        for record in seed_records.values():
            references.extend(record.get('referenced_works') or [])
        ids = [self.normalize_seed(ref) for ref in dict.fromkeys(references)]
        return self._fetch_by_ids([i for i in ids if i], desc='Backward citations')

    def search_queries(self, queries: Sequence[str],
                       year_min: Optional[int] = None,
                       year_max: Optional[int] = None) -> Dict[str, dict]:
        """Union of ``title_and_abstract.search`` queries.

        That filter is the filter spelling of the engine an ``api.query``
        hits, so an entry behaves like any OpenAlex search: its terms are
        AND-ed and stemmed (hyphens and plurals are ignored) unless the caller
        quotes them, in which case they must appear in that order — still
        stemmed, so no match is ever literal. ``AND`` / ``OR`` / ``NOT`` are
        honoured, and results come back relevance-ranked, so
        :attr:`max_per_query` truncates to the *best* matches.
        """
        found: Dict[str, dict] = {}
        for query in tqdm(queries, desc='Text queries',
                          disable=not self.show_progress or not queries):
            # A comma would be read as a filter separator — as a search term it
            # is meaningless anyway, so drop it. `|` (OR between filter values)
            # and a leading `!` (negation) are reserved too, but they are
            # legitimate operators here: leave them alone.
            filters = {'title_and_abstract.search': query.replace(',', ' ').strip()}
            filters.update(self._year_filters(year_min, year_max))
            found.update(self._search(filters, self.max_per_query))
        return found

    def to_dataframe(self, records: Iterable[dict]) -> pd.DataFrame:
        return from_openalex_records(records)

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _year_filters(year_min: Optional[int], year_max: Optional[int]) -> dict:
        """Year window as OpenAlex date filters (applied server-side)."""
        filters = {}
        if year_min is not None:
            filters['from_publication_date'] = f'{year_min}-01-01'
        if year_max is not None:
            filters['to_publication_date'] = f'{year_max}-12-31'
        return filters

    def _search(self, filters: dict, max_records: int) -> Dict[str, dict]:
        """One paginated /works query, keyed by work id."""
        result = self.client.search(filters=filters, max_records=max_records,
                                    show_progress=False)
        return {r['id']: r for r in result.records}

    def _fetch_by_ids(self, ids: Sequence[str], desc: str) -> Dict[str, dict]:
        """Fetch full works for normalized ids (``Wxxxx`` / ``doi:…``).

        Ids are batched into OR-ed ``openalex_id:`` / ``doi:`` filters — one
        request per :data:`_ID_CHUNK` ids. Ids OpenAlex does not know are
        simply absent from the result.
        """
        unique = [i for i in dict.fromkeys(ids) if i]
        work_ids = [i for i in unique if not i.startswith('doi:')]
        dois     = [i[len('doi:'):] for i in unique if i.startswith('doi:')]

        batches = [('openalex_id', work_ids[i:i + _ID_CHUNK])
                   for i in range(0, len(work_ids), _ID_CHUNK)]
        batches += [('doi', dois[i:i + _ID_CHUNK])
                    for i in range(0, len(dois), _ID_CHUNK)]

        found: Dict[str, dict] = {}
        for key, batch in tqdm(batches, desc=desc,
                               disable=not self.show_progress or not batches):
            page = self.client._fetch_page({'filter': f"{key}:" + '|'.join(batch),
                                            'per-page': len(batch)})
            for record in page.get('results', []):
                found[record['id']] = record
        return found
