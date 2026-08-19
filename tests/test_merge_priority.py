"""Tests for cross-source merge priority.

The source loaded first becomes the merge's primary dataset: its records win
duplicates and keep their IDs, and each dropped duplicate is aliased to the
surviving record (via ``cross_id_map``) so references still resolve. The
default priority puts OpenAlex first, so WoS duplicates are dropped in favour of
the OpenAlex record and references to a WoS duplicate resolve to the OpenAlex
key; WoS-only records survive and keep their WoS ids.
"""

import pandas as pd
import pytest

from pysyrev.bibdata import BibDataset
from pysyrev.core.bib import DEFAULT_FIELDS
from pysyrev.core.config import MergeConfig


def _dataset(rows):
    """Build a schema-valid BibDataset from ``[(id, doi, title), ...]``."""
    records = []
    for id_, doi, title in rows:
        rec = {col: pd.NA for col in DEFAULT_FIELDS}
        rec.update(id=id_, doi=doi, title=title, year=2020, cited_by=0)
        records.append(rec)
    df = pd.DataFrame(records)
    for col, kind in DEFAULT_FIELDS.items():
        df[col] = df[col].astype("string" if kind == "string" else "float")
    return BibDataset(bib_dataset=df)


@pytest.fixture
def open_alex():
    # OA1 duplicates WoS1 (same DOI); OA2 is OpenAlex-only.
    return _dataset([("OA1", "10.1000/a", "Shared paper on coupling networks"),
                     ("OA2", "10.1000/b", "OpenAlex only paper about topics")])


@pytest.fixture
def wos():
    # WOS1 duplicates OA1 (same DOI); WOS9 is WoS-only.
    return _dataset([("WOS1", "10.1000/a", "Shared paper on coupling networks"),
                     ("WOS9", "10.1000/z", "WoS only paper about references")])


class TestMergePriority:

    def test_openalex_first_wins_and_aliases_wos_duplicate(self, open_alex, wos):
        merged = open_alex.merge([wos])
        ids = set(merged.dataset["id"])
        # OpenAlex record kept, WoS duplicate dropped, WoS-only survives.
        assert ids == {"OA1", "OA2", "WOS9"}
        # Reference to the dropped WoS id resolves to the OpenAlex key.
        assert merged._cross_id_map == {"WOS1": "OA1"}

    def test_wos_first_reverses_the_direction(self, open_alex, wos):
        merged = wos.merge([open_alex])
        ids = set(merged.dataset["id"])
        assert ids == {"WOS1", "WOS9", "OA2"}
        assert merged._cross_id_map == {"OA1": "WOS1"}


class TestMergeConfigPriority:

    def test_default_puts_openalex_first(self):
        assert MergeConfig().priority == ["open_alex", "wos", "scopus", "pubmed"]

    def test_priority_is_overridable(self):
        assert MergeConfig(priority=["wos", "open_alex"]).priority == \
            ["wos", "open_alex"]
