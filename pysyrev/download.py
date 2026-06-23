"""
Full-text paper download module.

PaperDownloader retrieves PDFs or structured XML for a list of papers
identified by DOI.  Sources are tried in cascade order:

  1. Unpaywall  — legal open-access PDF discovered by DOI.
  2. OpenAlex   — open_access.oa_url from the OpenAlex Works API.
  3. Elsevier   — Text and Data Mining API (institutional key required).

Papers that cannot be retrieved from any source are flagged as ``manual``
in the download report so the user knows what still needs attention.

Usage
-----
  from pysyrev.core.config import DownloadConfig
  from pysyrev.download import PaperDownloader

  downloader = PaperDownloader.from_config("download_config.yaml")
  downloader.run().save()
"""

import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
import requests
from tqdm import tqdm

from pysyrev.core.config import DownloadConfig

# ── Column names (mirrors pysyrev.core.bib constants) ────────────────────────
_DOI_COL   = 'doi'
_TITLE_COL = 'title'

# ── Download-report column names and status values ────────────────────────────
_COL_DOI       = 'doi'
_COL_TITLE     = 'title'
_COL_STATUS    = 'status'
_COL_SOURCE    = 'source'
_COL_FILE_PATH = 'file_path'

_STATUS_DOWNLOADED = 'downloaded'
_STATUS_MANUAL     = 'manual'
_STATUS_NO_DOI     = 'no_doi'

_REPORT_COLS = [_COL_DOI, _COL_TITLE, _COL_STATUS, _COL_SOURCE, _COL_FILE_PATH]

# ── HTTP session shared across all download attempts ─────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({'User-Agent': 'pysyrev-downloader/1.0 (mailto:pysyrev@github.com)'})

_MIN_CONTENT_BYTES = 1_024   # responses smaller than 1 kB are treated as errors


# =============================================================================
# Internal helpers
# =============================================================================

def _sanitize_doi(doi: str) -> str:
    """Replace characters that are unsafe in file names with underscores."""
    return re.sub(r'[^\w\-]', '_', doi)


def _get(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[requests.Response]:
    """GET *url* and return the Response, or None on any error."""
    try:
        resp = _SESSION.get(url, headers=headers or {}, timeout=timeout)
        if resp.status_code == 200:
            return resp
    except requests.RequestException:
        pass
    return None


def _save(content: bytes, path: str) -> bool:
    """Write *content* to *path*. Returns False when content looks like an error page."""
    if len(content) < _MIN_CONTENT_BYTES:
        return False
    with open(path, 'wb') as fh:
        fh.write(content)
    return True


# =============================================================================
# Source-specific download functions
# =============================================================================

def _try_unpaywall(doi: str, email: str, dest: str, delay: float) -> Optional[str]:
    """Query Unpaywall for a legal OA PDF and download it.

    Unpaywall's fair-use policy requires an email and ≤ 1 req/s.
    """
    encoded = urllib.parse.quote(doi, safe='')
    url = f"https://api.unpaywall.org/v2/{encoded}?email={urllib.parse.quote(email)}"
    resp = _get(url)
    time.sleep(delay)
    if resp is None:
        return None

    data = resp.json()
    best = data.get('best_oa_location') or {}
    pdf_url = best.get('url_for_pdf')
    if not pdf_url:
        return None

    pdf_resp = _get(pdf_url)
    if pdf_resp and _save(pdf_resp.content, dest):
        return dest
    return None


def _try_openalex(doi: str, dest: str, delay: float, email: Optional[str] = None) -> Optional[str]:
    """Query the OpenAlex Works API for an OA URL and download the PDF.

    Uses the ``mailto`` polite-pool parameter when an email is provided.
    """
    encoded = urllib.parse.quote(doi, safe='')
    base = f"https://api.openalex.org/works/doi:{encoded}"
    url = f"{base}?mailto={urllib.parse.quote(email)}" if email else base

    resp = _get(url)
    time.sleep(delay)
    if resp is None:
        return None

    oa_url = (resp.json().get('open_access') or {}).get('oa_url')
    if not oa_url:
        return None

    pdf_resp = _get(oa_url)
    if pdf_resp and _save(pdf_resp.content, dest):
        return dest
    return None


def _try_elsevier(doi: str, api_key: str, fmt: str, dest: str, delay: float) -> Optional[str]:
    """Download a full-text article via the Elsevier TDM API.

    *fmt* is ``'xml'`` (structured full-text, recommended for LLM use) or
    ``'pdf'``.  Only works for Elsevier-published articles under an
    institutional TDM licence.
    """
    accept = 'text/xml' if fmt == 'xml' else 'application/pdf'
    encoded = urllib.parse.quote(doi, safe='')
    url = f"https://api.elsevier.com/content/article/doi/{encoded}"
    headers = {'X-ELS-APIKey': api_key, 'Accept': accept}

    resp = _get(url, headers=headers)
    time.sleep(delay)
    if resp and _save(resp.content, dest):
        return dest
    return None


# =============================================================================
# Result container
# =============================================================================

@dataclass
class _DownloadResult:
    rows: list = field(default_factory=list)

    def add(self, doi, title, status, source=None, file_path=None):
        self.rows.append({
            _COL_DOI:       doi,
            _COL_TITLE:     title,
            _COL_STATUS:    status,
            _COL_SOURCE:    source,
            _COL_FILE_PATH: file_path,
        })

    @property
    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=_REPORT_COLS)

    @property
    def n_downloaded(self) -> int:
        return sum(1 for r in self.rows if r[_COL_STATUS] == _STATUS_DOWNLOADED)

    @property
    def n_total(self) -> int:
        return len(self.rows)


# =============================================================================
# Public class
# =============================================================================

@dataclass
class PaperDownloader:
    """Downloads full-text papers using a cascade of open-access and
    publisher sources.

    Instantiate with :meth:`from_config`, then call :meth:`run` and
    :meth:`save`.

    Results are accessible via :attr:`report` (a ``pd.DataFrame``) at any
    time after :meth:`run` has been called.
    """
    config:       DownloadConfig
    _result:      _DownloadResult  = field(default_factory=_DownloadResult, init=False, repr=False)
    _last_source: Optional[str]    = field(default=None,                    init=False, repr=False)

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Union[str, DownloadConfig]) -> 'PaperDownloader':
        """Build a PaperDownloader from a YAML path or a DownloadConfig."""
        if isinstance(config, str):
            config = DownloadConfig.load(config)
        return cls(config=config)

    # ── Runtime ──────────────────────────────────────────────────────────────

    def run(self, dataset: Optional[pd.DataFrame] = None) -> 'PaperDownloader':
        """Run the download cascade on all papers in *dataset*.

        Parameters
        ----------
        dataset : pd.DataFrame, optional
            Must contain a ``doi`` column.  When *None*, loaded from
            ``config.doc_dataset``.

        Returns
        -------
        self
        """
        if dataset is None:
            dataset = pd.read_csv(self.config.doc_dataset)

        papers = dataset
        if self.config.max_papers is not None:
            papers = papers.head(self.config.max_papers)

        papers_dir   = os.path.join(self.config.output_dir, 'papers')
        elsevier_fmt = self.config.elsevier.format if self.config.elsevier else 'pdf'
        oa_email     = self.config.unpaywall.email if self.config.unpaywall else None

        result = _DownloadResult()

        for _, row in tqdm(papers.iterrows(), total=len(papers), desc="Downloading papers"):
            doi   = row.get(_DOI_COL)
            title = row.get(_TITLE_COL, '')

            if not doi or (isinstance(doi, float) and pd.isna(doi)):
                result.add(doi, title, _STATUS_NO_DOI)
                continue

            doi  = str(doi).strip()
            stem = _sanitize_doi(doi)

            # Skip papers already on disk (re-run safety).
            existing = self._find_existing(papers_dir, stem)
            if existing:
                result.add(doi, title, _STATUS_DOWNLOADED, source='cache', file_path=existing)
                continue

            file_path = self._cascade(doi, stem, papers_dir, elsevier_fmt, oa_email)

            if file_path:
                result.add(doi, title, _STATUS_DOWNLOADED, source=self._last_source, file_path=file_path)
            else:
                result.add(doi, title, _STATUS_MANUAL)

        self._result = result
        return self

    def save(self) -> 'PaperDownloader':
        """Write ``download_report.csv`` to ``output_dir`` and print a summary."""
        report_path = os.path.join(self.config.output_dir, 'download_report.csv')
        self._result.report.to_csv(report_path, index=False)
        n_ok    = self._result.n_downloaded
        n_total = self._result.n_total
        print(
            f"\nDownloaded {n_ok}/{n_total} papers. "
            f"{n_total - n_ok} require manual download "
            f"(see {report_path})."
        )
        return self

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def report(self) -> pd.DataFrame:
        """Download report as a DataFrame (available after :meth:`run`)."""
        return self._result.report

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _cascade(
        self,
        doi: str,
        stem: str,
        papers_dir: str,
        elsevier_fmt: str,
        oa_email: Optional[str],
    ) -> Optional[str]:
        """Try each source in order. Sets ``_last_source`` and returns the file path."""
        delay = self.config.request_delay

        # 1. Unpaywall
        if self.config.unpaywall:
            dest = os.path.join(papers_dir, f"{stem}.pdf")
            path = _try_unpaywall(doi, self.config.unpaywall.email, dest, delay)
            if path:
                self._last_source = 'unpaywall'
                return path

        # 2. OpenAlex
        dest = os.path.join(papers_dir, f"{stem}.pdf")
        path = _try_openalex(doi, dest, delay, email=oa_email)
        if path:
            self._last_source = 'openalex'
            return path

        # 3. Elsevier TDM
        if self.config.elsevier:
            dest = os.path.join(papers_dir, f"{stem}.{elsevier_fmt}")
            path = _try_elsevier(doi, self.config.elsevier.api_key, elsevier_fmt, dest, delay)
            if path:
                self._last_source = 'elsevier'
                return path

        self._last_source = None
        return None

    @staticmethod
    def _find_existing(papers_dir: str, stem: str) -> Optional[str]:
        """Return an existing file for *stem* (any extension), or None."""
        for ext in ('pdf', 'xml'):
            candidate = os.path.join(papers_dir, f"{stem}.{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None
