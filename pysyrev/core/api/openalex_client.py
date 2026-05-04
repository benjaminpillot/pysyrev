"""
Low-level OpenAlex API client.

Responsibilities:
  - Run a query against the /works endpoint, with full-text search
    (`search=`) and/or structured filters (`filter=`).
  - Iterate the cursor-based pagination automatically.
  - Retry transient errors (5xx, 429) with exponential backoff.
  - Resume mid-pagination when retries fail, so a partial run can be
    completed without re-fetching everything.
  - Surface the raw JSON of each work. Mapping to the project schema
    (`DEFAULT_FIELDS`) is the responsibility of a higher layer.

Notes:
  - No authentication required. Optionally accept an email for the polite
    pool, which gives more generous rate limits (100k req/day).
  - Cursor-based pagination scales to arbitrary result sizes (no 10k cap
    like the offset-based pagination).

Reference:
  https://docs.openalex.org/
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import requests
from tqdm import tqdm


_OPENALEX_BASE_URL = 'https://api.openalex.org/works'

# Max items per page (OpenAlex's documented hard cap).
_PAGE_SIZE = 200

# Retry policy (mirrors WosClient).
_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 2.0   # doubles on each retry: 2, 4, 8, 16, 32

# Client-side throttling. OpenAlex recommends <= 10 req/s.
_MIN_INTERVAL_SECONDS = 0.1


@dataclass
class OpenAlexSearchResult:
    """Result of one search() call."""
    total_found: int
    records: List[dict]


class OpenAlexClient:
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
        self.api_key = api_key
        self.email = email
        self.base_url = base_url.rstrip('/')
        self.session_file = Path(session_file) if session_file else None

        self._session = requests.Session()
        ua = 'pysyrev'
        if email:
            ua = f'pysyrev (mailto:{email})'
        self._session.headers.update({
            'Authorization' : f"Bearer {self.api_key}",
            'User-Agent': ua,
            'Accept': 'application/json',
        })

        self._last_request_at = 0.0

    # ---- single-page primitive ----------------------------------------------

    def _fetch_page(self, params: dict) -> dict:
        """Fetch one page. `params` must already carry the search/filter/cursor.
        Returns the raw JSON body."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _MIN_INTERVAL_SECONDS:
            time.sleep(_MIN_INTERVAL_SECONDS - elapsed)

        # Polite-pool credit can also be passed as a query param when you
        # cannot set headers (e.g. some proxies strip them).
        full_params = dict(params)
        if self.email and 'mailto' not in full_params:
            full_params['mailto'] = self.email

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._session.get(self.base_url, params=full_params, timeout=60)
                self._last_request_at = time.monotonic()

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f'Retryable HTTP {response.status_code}: {response.text[:200]}',
                        response=response,
                    )
                response.raise_for_status()
                return response.json()

            except (requests.RequestException, requests.HTTPError) as e:
                last_error = e
                if attempt == _MAX_RETRIES - 1:
                    break
                backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)

        raise RuntimeError(
            f'OpenAlex request failed after {_MAX_RETRIES} retries: {last_error}'
        ) from last_error

    # ---- session state for resume -------------------------------------------

    @staticmethod
    def _session_key(query: Optional[str], filters: Optional[dict]) -> str:
        """Stable identity for a (query, filters) pair, used to detect when
        a saved session refers to a different request."""
        return json.dumps({'q': query, 'f': filters or {}}, sort_keys=True)

    def _load_session(self, key: str) -> dict:
        if not self.session_file or not self.session_file.exists():
            return {'key': key, 'records': [], 'cursor': '*', 'total_found': None}
        with open(self.session_file, 'r') as fh:
            state = json.load(fh)
        if state.get('key') != key:
            return {'key': key, 'records': [], 'cursor': '*', 'total_found': None}
        return state

    def _save_session(self, state: dict) -> None:
        if not self.session_file:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, 'w') as fh:
            json.dump(state, fh)

    def _clear_session(self) -> None:
        if self.session_file and self.session_file.exists():
            self.session_file.unlink()

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
