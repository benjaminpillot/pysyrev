"""
Low-level Web of Science (Expanded API) client.

Responsibilities:
  - Authenticate via the X-ApiKey header.
  - Run a query in WoS Query Language and iterate the paginated results.
  - Surface the raw JSON of each record. Mapping to the project schema
    (`DEFAULT_FIELDS`) is the responsibility of a higher layer.

Throttling, retry/backoff and resume-after-interrupt are shared with the other
REST clients — see :class:`pysyrev.core.api.base.PaginatedClient`.

Reference:
  https://developer.clarivate.com/apis/wos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tqdm import tqdm

from pysyrev.core.api.base import PaginatedClient


# WoS Expanded API base URL. The Starter API has a different URL scheme;
# adjust here if you switch tier.
_WOS_BASE_URL = 'https://wos-api.clarivate.com/api/wos'

# How many records per page. The API caps this; 100 is the documented max
# for Expanded.
_PAGE_SIZE = 100


@dataclass
class WosSearchResult:
    """Result of one search() call."""
    total_found: int
    records: List[dict]


class WosClient(PaginatedClient):
    """Low-level client for WoS Expanded API.

    Usage:
        client = WosClient(api_key='abc123')
        result = client.search('TS=("agent-based") AND PY=2015-2024')
        for record in result.records:
            ...
    """

    name = 'WoS API'
    # Rate limiting (client-side guard rail). Expanded typically allows 5 req/s.
    min_interval = 0.25
    resume_fields = {'next_first_record': 1}

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
        super().__init__(base_url, session_file)
        self.api_key = api_key
        self.database = database
        self._session.headers.update({
            'X-ApiKey': self.api_key,
            'Accept': 'application/json',
        })

    def _fetch_page(self, query: str, first_record: int, count: int) -> dict:
        """Fetch a single page (max `count` records starting at `first_record`)."""
        return self._get({
            'databaseId':  self.database,
            'usrQuery':    query,
            'count':       count,
            'firstRecord': first_record,
        })

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
                    'key': query,
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
