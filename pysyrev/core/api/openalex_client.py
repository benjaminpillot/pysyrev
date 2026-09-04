"""
Low-level OpenAlex API client.

Responsibilities:
  - Run a query against the /works endpoint, with full-text search
    (`search=`) and/or structured filters (`filter=`).
  - Iterate the cursor-based pagination automatically.
  - Surface the raw JSON of each work. Mapping to the project schema
    (`DEFAULT_FIELDS`) is the responsibility of a higher layer.

Throttling, retry/backoff and resume-after-interrupt are shared with the other
REST clients — see :class:`pysyrev.core.api.base.PaginatedClient`.

Notes:
  - Optionally accept an email for the polite pool, which gives more
    generous rate limits (100k req/day).
  - Cursor-based pagination scales to arbitrary result sizes (no 10k cap
    like the offset-based pagination).

Reference:
  https://docs.openalex.org/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Union

from tqdm import tqdm

from pysyrev.core.api.base import PaginatedClient


_OPENALEX_BASE_URL = 'https://api.openalex.org/works'

# Max items per page (OpenAlex's documented hard cap).
_PAGE_SIZE = 200


@dataclass
class OpenAlexSearchResult:
    """Result of one search() call."""
    total_found: int
    records: List[dict]


class OpenAlexClient(PaginatedClient):
    """Low-level client for OpenAlex /works.

    Usage:
        client = OpenAlexClient(email='you@example.com')

        # Full-text search (BM25 over title+abstract):
        result = client.search(query='agent-based energy transition',
                               filters={'publication_year': '2015-2024',
                                        'type': 'article'})

        # Or pure structured filters:
        result = client.search(filters={'concepts.id': 'C42475967',
                                        'publication_year': '2020-2024'})

        for work in result.records:
            ...
    """

    name = 'OpenAlex'
    # Client-side throttling. OpenAlex recommends <= 10 req/s.
    min_interval = 0.1
    resume_fields = {'cursor': '*'}

    def __init__(self,
                 api_key: str,
                 email: Union[None, str] = None,
                 base_url: str = _OPENALEX_BASE_URL,
                 session_file: Union[None, str] = None):
        """
        Parameters
        ----------
        api_key : str
            OpenAlex API key
        email : Union[None, str]
            Email used to opt into the OpenAlex 'polite pool'. Strongly
            recommended for any non-trivial usage; gives more generous
            rate limits and helps Clarivate contact you in case of issues.
        base_url : str
            Override the default URL. Mostly useful for testing.
        session_file : Union[None, str]
            Path to a JSON file storing pagination state. When provided,
            an interrupted run can be resumed from the last saved cursor.
        """
        super().__init__(base_url, session_file)
        self.api_key = api_key
        self.email = email
        self._session.headers.update({
            'Authorization': f"Bearer {self.api_key}",
            'User-Agent': f'pysyrev (mailto:{email})' if email else 'pysyrev',
            'Accept': 'application/json',
        })

    def _fetch_page(self, params: dict) -> dict:
        """Fetch one page. `params` must already carry the search/filter/cursor."""
        # Polite-pool credit can also be passed as a query param when you
        # cannot set headers (e.g. some proxies strip them).
        full_params = dict(params)
        if self.email and 'mailto' not in full_params:
            full_params['mailto'] = self.email
        return self._get(full_params)

    # ---- session state for resume -------------------------------------------

    @staticmethod
    def _session_key(query: Optional[str], filters: Optional[dict]) -> str:
        """Stable identity for a (query, filters) pair, used to detect when
        a saved session refers to a different request."""
        return json.dumps({'q': query, 'f': filters or {}}, sort_keys=True)

    # ---- public API ---------------------------------------------------------

    def search(self,
               query: Union[None, str] = None,
               search_param: Union[None, str] = "search.title_and_abstract",
               filters: Optional[dict] = None,
               max_records: Optional[int] = None,
               show_progress: bool = True) -> OpenAlexSearchResult:
        """Run a search and return all matching works (paginated).

        Parameters
        ----------
        query : Union[None, str]
            Free-text search (BM25 over title + abstract). Use it as you
            would the OpenAlex search bar.
        search_param : Union[None, str]
            Search query parameter
            (e.g. : search, search.exact, search.title_and_abstract, etc. see OA API doc)
        filters : Union[None, dict]
            Structured filters. Keys are OpenAlex filter names, values are
            the filter values. Examples:
              {'publication_year': '2020-2024'}
              {'type': 'article'}
              {'concepts.id': 'C42475967'}
              {'authorships.institutions.country_code': 'FR'}
            Combined with AND server-side. See OpenAlex docs for the full list.
        max_records : Union[None, int]
            Cap on the number of records to retrieve. None = all.
        show_progress : bool
            Display a tqdm progress bar over pages.

        Returns
        -------
        OpenAlexSearchResult
            With `total_found` and the actual list of records retrieved.

        At least one of `query` or `filters` must be set.
        """
        if not query and not filters:
            raise ValueError("At least one of `query` or `filters` must be provided.")

        key = self._session_key(query, filters)
        state = self._load_session(key)
        records = list(state['records'])
        cursor = state['cursor']
        total_found = state['total_found']

        # Build the static portion of the request params.
        base_params = {'per-page': _PAGE_SIZE}
        if query:
            base_params[search_param] = query
        if filters:
            # OpenAlex expects a comma-separated `key:value` string for filters.
            base_params['filter'] = ','.join(f'{k}:{v}' for k, v in filters.items())

        # Loop until cursor exhausted or max_records reached.
        progress = None
        try:
            while True:
                if max_records is not None and len(records) >= max_records:
                    break

                params = dict(base_params)
                params['cursor'] = cursor

                page = self._fetch_page(params)

                # Capture total_found from the very first page (and verify
                # it stays consistent on subsequent pages).
                page_total = page.get('meta', {}).get('count', 0)
                if total_found is None:
                    total_found = page_total
                    target = total_found if max_records is None else min(total_found, max_records)
                    if show_progress:
                        progress = tqdm(total=target, initial=len(records), desc='OpenAlex pages')

                page_records = page.get('results', [])
                if not page_records:
                    break

                records.extend(page_records)
                if progress is not None:
                    progress.update(len(page_records))

                next_cursor = page.get('meta', {}).get('next_cursor')
                if not next_cursor:
                    break
                cursor = next_cursor

                # Persist after every successful page so we can resume on crash.
                self._save_session({
                    'key': key,
                    'records': records,
                    'cursor': cursor,
                    'total_found': total_found,
                })

        finally:
            if progress is not None:
                progress.close()

        if max_records is not None:
            records = records[:max_records]

        # Search completed -> session no longer needed.
        self._clear_session()

        return OpenAlexSearchResult(
            total_found=total_found if total_found is not None else 0,
            records=records,
        )
