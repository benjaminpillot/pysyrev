"""Tests for corpus-level abstract completion (bib.abstract_completion)."""

import pandas as pd
import pytest

from pysyrev.bibdata import BibDataset
from pysyrev.core.bib import DEFAULT_FIELDS
from pysyrev.core.config import BibConfig, AbstractCompletionConfig


def _valid_df():
    """A minimal DEFAULT_FIELDS-conforming dataset (one row)."""
    row = {col: "x" for col in DEFAULT_FIELDS}
    df = pd.DataFrame([row])
    df["year"] = [2020]
    df["cited_by"] = [1]
    return df


class TestAbstractCompletionConfig:

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="provider"):
            AbstractCompletionConfig(api_key="k")

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            AbstractCompletionConfig(provider="wos")

    def test_valid(self):
        cfg = AbstractCompletionConfig(provider="wos", api_key="k", cache_dir="/c")
        assert (cfg.provider, cfg.api_key, cfg.cache_dir) == ("wos", "k", "/c")


class TestBibConfigParsing:

    def test_parses_completion_list_of_dicts(self):
        cfg = BibConfig(abstract_completion=[
            {"provider": "wos", "api_key": "k1"},
            {"provider": "scopus", "api_key": "k2"},
        ])
        assert all(isinstance(c, AbstractCompletionConfig)
                   for c in cfg.abstract_completion)
        assert [c.provider for c in cfg.abstract_completion] == ["wos", "scopus"]

    def test_absent_completion_stays_none(self):
        assert BibConfig(open_alex="/x.csv").abstract_completion is None

    def test_openalex_no_longer_accepts_wos_completion(self):
        # The completion sub-block moved up to bib.abstract_completion; declaring
        # it under the source is now an unknown field.
        with pytest.raises(TypeError):
            BibConfig(open_alex={"source": "file", "file": "/x.csv",
                                 "wos_completion": {"api_key": "k"}})


class TestApplyCompletions:

    def test_dispatches_providers_in_order(self, monkeypatch):
        calls = []

        def make_factory():
            def factory(cfg):
                def completer(dataframe, show_progress):
                    calls.append(cfg.provider)
                    return 0
                return completer
            return factory

        monkeypatch.setattr(BibDataset, "_COMPLETERS", {
            "wos":    ("WoS", make_factory()),
            "scopus": ("Scopus", make_factory()),
        })
        ds = BibDataset(bib_dataset=_valid_df())
        ds.apply_completions([
            AbstractCompletionConfig(provider="wos", api_key="k"),
            AbstractCompletionConfig(provider="scopus", api_key="k"),
        ], verbose=False)
        assert calls == ["wos", "scopus"]

    def test_unknown_provider_raises(self):
        ds = BibDataset(bib_dataset=_valid_df())
        with pytest.raises(ValueError, match="Unknown completion provider"):
            ds.apply_completions(
                [AbstractCompletionConfig(provider="nope", api_key="k")], verbose=False)

    def test_empty_list_is_noop(self):
        ds = BibDataset(bib_dataset=_valid_df())
        assert ds.apply_completions([], verbose=False) is ds
        assert ds.apply_completions(None, verbose=False) is ds
