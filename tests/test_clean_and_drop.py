"""Tests for the no-abstract drop of clean_and_drop (bib.clean).

A search-built corpus wants the drop: a record with no abstract cannot be
screened. A closed corpus — a DOI list of papers already chosen — wants the
opposite: every declared paper must survive the pipeline, abstract or not.
``clean.drop_record_with_no_abstract`` is the switch between the two.
"""

import numpy as np
import pandas as pd
import pytest

from pysyrev.bibdata import BibDataset, OpenAlexDataset
from pysyrev.core.bib import DEFAULT_FIELDS
from pysyrev.core.config import BibConfig, CleanConfig


_REAL_ABSTRACT = (
    "This paper presents an agent-based model of the energy transition. "
    "We show that the adoption of solar panels depends on peer effects, and "
    "we discuss the policy implications of these results."
)


def _df(abstracts):
    """A DEFAULT_FIELDS dataset, one row per entry of *abstracts*."""
    rows = [{col: "x" for col in DEFAULT_FIELDS} for _ in abstracts]
    df = pd.DataFrame(rows)
    df["abstract"] = abstracts
    df["doi"] = [f"10.1000/{i}" for i in range(len(abstracts))]
    df["year"] = 2020
    df["cited_by"] = 1
    return df


class TestDropRecordWithNoAbstract:

    def test_drops_by_default(self):
        ds = BibDataset(bib_dataset=_df([_REAL_ABSTRACT, np.nan]))
        ds.clean_and_drop()
        assert len(ds.dataset) == 1
        assert ds.dataset.doi.tolist() == ["10.1000/0"]

    def test_keeps_records_when_disabled(self):
        ds = BibDataset(bib_dataset=_df([_REAL_ABSTRACT, np.nan]))
        ds.clean_and_drop(drop_record_with_no_abstract=False)
        assert len(ds.dataset) == 2
        assert ds.dataset.doi.tolist() == ["10.1000/0", "10.1000/1"]
        assert pd.isna(ds.dataset.abstract.iloc[1])

    def test_keeps_the_record_whose_abstract_the_filter_rejected(self):
        # 'no abstract available' is known garbage: cleaning blanks it out, so
        # the record only survives because the drop is off.
        ds = BibDataset(bib_dataset=_df([_REAL_ABSTRACT, "No abstract available"]))
        ds.clean_and_drop(drop_record_with_no_abstract=False)
        assert len(ds.dataset) == 2
        assert pd.isna(ds.dataset.abstract.iloc[1])

    def test_cleaning_still_runs_when_the_drop_is_off(self):
        ds = BibDataset(bib_dataset=_df([_REAL_ABSTRACT]))
        ds.dataset.loc[0, "doi"] = "10.1000/abc [suppl]"
        ds.clean_and_drop(drop_record_with_no_abstract=False,
                          max_abstract_chars=40)
        assert ds.dataset.doi.iloc[0] == "10.1000/abc"   # cut at the first invalid char
        assert len(ds.dataset.abstract.iloc[0]) <= 40


class TestCleanConfig:

    def test_defaults_to_dropping(self):
        assert CleanConfig().drop_record_with_no_abstract is True
        assert BibConfig(open_alex="/x.csv").clean.drop_record_with_no_abstract is True

    def test_parsed_from_the_clean_block(self):
        cfg = BibConfig(open_alex="/x.csv",
                        clean={"drop_record_with_no_abstract": False})
        assert cfg.clean.drop_record_with_no_abstract is False


class TestFromConfigPlumbing:

    @pytest.mark.parametrize("drop, expected", [(True, 1), (False, 2)])
    def test_config_flag_reaches_clean_and_drop(self, monkeypatch, drop, expected):
        source = OpenAlexDataset(bib_dataset=_df([_REAL_ABSTRACT, np.nan]))
        monkeypatch.setattr(OpenAlexDataset, "from_config",
                            classmethod(lambda cls, cfg: source))

        merged = BibDataset.from_config(
            BibConfig(open_alex="/x.csv",
                      clean={"drop_record_with_no_abstract": drop})
        )
        assert len(merged.dataset) == expected
