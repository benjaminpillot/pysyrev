"""External API clients for bibliographic sources."""

from pysyrev.core.api.openalex_client import OpenAlexClient, OpenAlexSearchResult
from pysyrev.core.api.wos_client import WosClient, WosSearchResult


__all__ = [
    'OpenAlexClient',
    'OpenAlexSearchResult',
    'WosClient',
    'WosSearchResult',
]
