"""Seed-driven corpus expansion: seeds -> candidate pool.

:mod:`~pysyrev.core.seed_expansion.base` holds the provider-agnostic machinery
(seed files, arm orchestration, deduplication); one module per provider holds
the arms themselves. OpenAlex is the only backend today — a new one implements
:class:`SeedExpander` and registers here under its provider name.
"""

from pysyrev.core.seed_expansion.base import (SeedExpander, SeedExpansionResult,
                                              dedup_pool, normalize_doi,
                                              read_seed_file, title_key)
from pysyrev.core.seed_expansion.openalex import OpenAlexExpander

# Provider name -> expander class, as used by `source: seed` config blocks.
EXPANDERS = {
    OpenAlexExpander.name: OpenAlexExpander,
}

__all__ = [
    'EXPANDERS',
    'OpenAlexExpander',
    'SeedExpander',
    'SeedExpansionResult',
    'dedup_pool',
    'normalize_doi',
    'read_seed_file',
    'title_key',
]
