"""
Low-level Web of Science (Expanded API) client.

Responsibilities:
  - Authenticate via the X-ApiKey header.
  - Run a query in WoS Query Language and iterate the paginated results.
  - Retry transient errors (5xx, 429) with exponential backoff.
  - Resume mid-pagination when retries fail, so a partial run can be
    completed without re-fetching everything.
  - Surface the raw JSON of each record. Mapping to the project schema
    (`DEFAULT_FIELDS`) is the responsibility of a higher layer.

Not done here (on purpose):
  - Mapping to the project DataFrame schema -> see wos_search.py.
  - Caching to disk -> see cache.py.

Reference:
  https://developer.clarivate.com/apis/wos
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import requests
from tqdm import tqdm


# WoS Expanded API base URL. The Starter API has a different URL scheme;
# adjust here if you switch tier.
_WOS_BASE_URL = 'https://wos-api.clarivate.com/api/wos'

# How many records per page. The API caps this; 100 is the documented max
# for Expanded.
_PAGE_SIZE = 100

# Retry policy.
_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 2.0   # doubles on each retry: 2, 4, 8, 16, 32

# Rate limiting (client-side guard rail). Expanded typically allows 5 req/s.
_MIN_INTERVAL_SECONDS = 0.25


@dataclass
class WosSearchResult:
    """Result of one search() call."""
    total_found: int
    records: List[dict]


class WosClient:
    """Low-level client for WoS Expanded API.

    Usage:
        client = WosClient(api_key='abc123')
        result = client.search('TS=("agent-based") AND PY=2015-2024')
        for record in result.records:
            ...
    """

    def __init__(self,
                 api_key: str,
                 database: str = 'WOS',
                 base_url: str = _WOS_BASE_URL,
                 session_file: Optional[str] = None):
        """
        Parameters
        ----------
        api_key : str
            The Clarivate WoS Expanded API key.
        database : str, default 'WOS'
            Which database to query. 'WOS' is the Web of Science Core Collection.
        base_url : str
            Base URL of the WoS Expanded API. Override only if you know
            you are on a different tier (Lite/Starter use other URLs).
        session_file : Union[None, str]
            Path to a JSON file storing pagination state. When provided,
            an interrupted run can be resumed from the last saved page.
        """
        self.api_key = api_key
        self.database = database
        self.base_url = base_url.rstrip('/')
        self.session_file = Path(session_file) if session_file else None

        self._session = requests.Session()
        self._session.headers.update({
            'X-ApiKey': self.api_key,
            'Accept': 'application/json',
        })

        self._last_request_at = 0.0  # monotonic-style throttling

    # ---- single-page primitive ----------------------------------------------

    def _fetch_page(self, query: str, first_record: int, count: int) -> dict:
        """Fetch a single page (max `count` records starting at `first_record`).
        Returns the raw JSON body."""
        # Client-side throttling.
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _MIN_INTERVAL_SECONDS:
            time.sleep(_MIN_INTERVAL_SECONDS - elapsed)

        params = {
            'databaseId': self.database,
            'usrQuery': query,
            'count': count,
            'firstRecord': first_record,
        }

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._session.get(self.base_url, params=params, timeout=60)
                self._last_request_at = time.monotonic()

                # 429 (rate limit) and 5xx (server side): retryable.
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f'Retryable HTTP {response.status_code}: {response.text[:200]}',
                        response=response,
                    )
                response.raise_for_status()
                return response.json()

            except requests.HTTPError as e:
                # Re-raise non-retryable client errors (4xx != 429) immediately.
                sc = e.response.status_code if e.response is not None else 0
                if sc != 429 and sc < 500:
                    raise
                last_error = e
                if attempt == _MAX_RETRIES - 1:
                    break
                backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)
            except requests.RequestException as e:
                last_error = e
                if attempt == _MAX_RETRIES - 1:
                    break
                backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)

        raise RuntimeError(
            f'WoS API request failed after {_MAX_RETRIES} retries: {last_error}'
        ) from last_error

    # ---- session state for resume -------------------------------------------

    def _load_session(self, query: str) -> dict:
        """Read existing session state for this query, if any."""
        if not self.session_file or not self.session_file.exists():
            return {'query': query, 'records': [], 'next_first_record': 1, 'total_found': None}
        with open(self.session_file, 'r') as fh:
            state = json.load(fh)
        if state.get('query') != query:
            # Different query than what was saved; ignore the stale session.
            return {'query': query, 'records': [], 'next_first_record': 1, 'total_found': None}
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
               query: str,
               max_records: Optional[int] = None,
               show_progress: bool = True) -> WosSearchResult:
        """Run a search and return all matching records (paginated).

        Parameters
        ----------
        query : str
            WoS Query Language string, e.g. 'TS=("machine learning") AND PY=2020-2024'.
        max_records : Union[int, None]
            Hard cap on the number of records to retrieve. None = all.
        show_progress : bool
            Display a tqdm progress bar over pages.

        Returns
        -------
        WosSearchResult
            With `total_found` (the total reported by the API) and
            `records` (the actual list of record dicts retrieved).
        """
        state = self._load_session(query)
        records = list(state['records'])
        first_record = state['next_first_record']
        total_found = state['total_found']

        # Initial probe to learn `total_found` if we don't have it yet.
        if total_found is None:
            page = self._fetch_page(query, first_record=first_record, count=_PAGE_SIZE)
            total_found = self._extract_total_found(page)
            page_records = self._extract_records(page)
            records.extend(page_records)
            first_record += len(page_records)

        target = total_found if max_records is None else min(total_found, max_records)
        progress = tqdm(total=target, initial=len(records), desc='WoS pages') if show_progress else None

        try:
            while len(records) < target:
                page = self._fetch_page(query, first_record=first_record, count=_PAGE_SIZE)
                page_records = self._extract_records(page)
                if not page_records:
                    break  # API stopped returning rows; safe to exit.
                records.extend(page_records)
                first_record += len(page_records)
                if progress is not None:
                    progress.update(len(page_records))

                # Persist state after every successful page so an interrupt
                # (Ctrl+C, network outage, kernel crash) is recoverable.
                self._save_session({
                    'query': query,
                    'records': records,
                    'next_first_record': first_record,
                    'total_found': total_found,
                })

        finally:
            if progress is not None:
                progress.close()

        # Truncate to max_records if the caller asked for less than total.
        if max_records is not None:
            records = records[:max_records]

        # Search completed successfully -> session no longer needed.
        self._clear_session()

        return WosSearchResult(total_found=total_found, records=records)

    # ---- helpers to navigate the WoS JSON shape -----------------------------

    @staticmethod
    def _extract_total_found(page: dict) -> int:
        """Pull the total record count from a WoS Expanded response."""
        try:
            return int(page['QueryResult']['RecordsFound'])
        except (KeyError, TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_records(page: dict) -> List[dict]:
        """Pull the list of records from a WoS Expanded response page."""
        try:
            recs = page['Data']['Records']['records']['REC']
        except (KeyError, TypeError):
            return []
        # The API returns either a single dict or a list depending on count.
        if isinstance(recs, dict):
            return [recs]
        return list(recs)
