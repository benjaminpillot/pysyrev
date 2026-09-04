"""
Shared machinery for the bibliographic REST clients.

Both WoS and OpenAlex are the same kind of thing: a rate-limited, paginated,
retrying GET against one endpoint, with the pagination state persisted so an
interrupted run can be resumed instead of re-fetched. Only the query grammar,
the pagination cursor and the JSON shape differ — those stay in the subclasses;
everything below is identical and lives here once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests


# Retry policy, shared by every client.
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0   # doubles on each retry: 2, 4, 8, 16, 32


class PaginatedClient:
    """Base for a throttled, resumable, retrying JSON client.

    Subclasses set :attr:`name` (used in error messages), :attr:`min_interval`
    (client-side rate guard, in seconds) and :attr:`resume_fields` (the extra
    pagination state their :meth:`search` needs to carry across a resume), then
    call :meth:`_get` for each page.
    """

    name = 'API'
    min_interval = 0.25
    resume_fields: dict = {}

    def __init__(self, base_url: str, session_file: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session_file = Path(session_file) if session_file else None

        self._session = requests.Session()
        self._last_request_at = 0.0  # monotonic-style throttling

    # ---- single-page primitive ----------------------------------------------

    def _get(self, params: dict) -> dict:
        """GET one page and return the raw JSON body.

        Retries 429 and 5xx with exponential backoff; any other 4xx is the
        caller's fault (a malformed query, a bad key) and is raised at once
        rather than repeated five times.
        """
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        last_error = None
        for attempt in range(MAX_RETRIES):
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
                sc = e.response.status_code if e.response is not None else 0
                if sc != 429 and sc < 500:
                    raise
                last_error = e
            except requests.RequestException as e:
                last_error = e

            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))

        raise RuntimeError(
            f'{self.name} request failed after {MAX_RETRIES} retries: {last_error}'
        ) from last_error

    # ---- session state for resume -------------------------------------------

    def _load_session(self, key: str) -> dict:
        """Saved pagination state for *key*, or a blank one.

        A session saved under a different key describes a different request and
        is ignored rather than resumed.
        """
        blank = {'key': key, 'records': [], 'total_found': None,
                 **self.resume_fields}
        if not self.session_file or not self.session_file.exists():
            return blank
        with open(self.session_file, 'r') as fh:
            state = json.load(fh)
        return state if state.get('key') == key else blank

    def _save_session(self, state: dict) -> None:
        if not self.session_file:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, 'w') as fh:
            json.dump(state, fh)

    def _clear_session(self) -> None:
        if self.session_file and self.session_file.exists():
            self.session_file.unlink()
