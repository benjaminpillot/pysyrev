"""Old-style DOIs must survive the pipeline whole.

SICI DOIs — ``10.1002/(SICI)1096-9845(199908)28:8<879::AID-EQE845>3.0.CO;2-1``
and friends, minted through the early 2000s — carry parentheses, angle brackets,
``;`` and ``#``. Two passes used to cut them short: `clean_doi`, which stopped at
the first character outside a too-narrow whitelist, and the reference splitter,
which broke a cell on a bare ``;``.
"""

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.bib import split_references
from pysyrev.core.clean import clean_doi
from pysyrev.core.networks import reference_sets
from pysyrev.core.references import add_reference_keys, has_unkeyed_references


_SICI = "10.1002/(SICI)1096-9845(199908)28:8<879::AID-EQE845>3.0.CO;2-1"


class TestCleanDoi:

    @pytest.mark.parametrize("raw", [
        _SICI,
        "10.1002/1521-3773(20010316)40:6<987::AID-ANIE9870>3.3.CO;2-#",
        "10.1016/S0006-3207(00)00042-9",
        "https://doi.org/" + _SICI,
    ])
    def test_keeps_the_doi_whole(self, raw):
        assert clean_doi(raw) == raw

    @pytest.mark.parametrize("raw, expected", [
        ("10.1000/abc [in Chinese]", "10.1000/abc"),   # a note after the DOI
        ("10.1000/abc, 10.1000/def", "10.1000/abc"),   # two DOIs in one cell
        ("10.1000/abc; 10.1000/def", "10.1000/abc"),
        ("10.1000/abc.", "10.1000/abc"),               # sentence punctuation
    ])
    def test_drops_what_trails_the_doi(self, raw, expected):
        assert clean_doi(raw) == expected

    def test_non_string_passes_through(self):
        assert clean_doi(np.nan) is np.nan


class TestSplitReferences:

    def test_a_sici_doi_is_one_token(self):
        assert split_references(f"{_SICI}; W2222") == [_SICI, "W2222"]

    def test_ignores_a_trailing_separator(self):
        assert split_references("W1; W2;") == ["W1", "W2"]

    @pytest.mark.parametrize("value", [None, np.nan, "", "   ", 42])
    def test_empty_for_a_cell_holding_nothing(self, value):
        assert split_references(value) == []


class TestReferencesCarryingSiciDois:
    """The splitter's callers, on a WoS reference string ending in a SICI DOI."""

    def _df(self):
        ref = f"Smith J, 1999, EARTHQ ENG STRUCT D, V28, P879, DOI {_SICI}"
        return pd.DataFrame({"id": ["W1"], "references": [f"{ref}; W2222"]})

    def test_reference_keys_keep_the_doi_whole(self):
        keys = add_reference_keys(self._df(), {})["reference_keys"].iloc[0]
        assert keys == f"{_SICI.lower()}; W2222"

    def test_the_doi_fragment_is_not_a_reference_of_its_own(self):
        _, refsets = reference_sets(
            pd.DataFrame({"id": ["W1"], "references": [f"{_SICI}; W2222"]}))
        assert refsets == [{_SICI, "W2222"}]

    def test_still_reports_raw_references_as_unkeyed(self):
        assert has_unkeyed_references(self._df()) is True
