"""Tests for canonical reference keys (a source-agnostic coupling basis).

Each reference is remapped to its normalized DOI when derivable, else its native
token is kept. The Wxxxx->DOI table is seeded for free from the corpus rows,
extended from a persistent CSV cache, and — for extra-corpus OpenAlex ids —
resolved once via the OpenAlex API. The key property is that a WoS reference and
an OpenAlex reference to the same work collapse onto the same DOI, independent of
merge order.
"""

import os

import pandas as pd
import pytest

from pysyrev.core import references as R


class TestTokenNormalization:

    @pytest.mark.parametrize("token, expected", [
        ("https://openalex.org/W123", "W123"),
        ("W123", "W123"),
        ("w123", "W123"),
        ("Smith J, 2019, NATURE", None),
        ("10.1000/abc", None),
        (None, None),
    ])
    def test_as_openalex_id(self, token, expected):
        assert R._as_openalex_id(token) == expected

    @pytest.mark.parametrize("doi, expected", [
        ("https://doi.org/10.1000/ABC", "10.1000/abc"),
        ("10.1000/abc", "10.1000/abc"),
        ("  10.1000/abc.,;", "10.1000/abc"),
        ("", None),
        (None, None),
    ])
    def test_normalize_doi_value(self, doi, expected):
        assert R._normalize_doi_value(doi) == expected


class TestSeedTable:

    def test_seeds_every_source_row_with_doi(self):
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1", "https://openalex.org/W2", "WOS:000123"],
            "doi": ["https://doi.org/10.1000/AAA", None, "10.1000/ccc"],
        })
        # Agnostic seed: every row with a DOI is keyed by its id (OpenAlex ids
        # normalized to bare Wxxxx, other schemes kept verbatim). W2 has no DOI.
        assert R.seed_table_from_dataframe(df) == {
            "W1": "10.1000/aaa", "WOS:000123": "10.1000/ccc"}


class TestCanonicalMapping:

    def test_four_token_cases_and_fallback(self):
        table = {"W1": "10.1000/aaa"}  # W1 known, W9 unknown
        df = pd.DataFrame({
            "references": [
                "https://openalex.org/W1; https://openalex.org/W9",   # known -> DOI, unknown -> kept
                "Smith J, 2019, NATURE, V1, P1, DOI 10.1000/bbb",     # WoS raw with DOI
                "10.1000/CCC; total garbage",                          # bare DOI + junk kept
            ],
        })
        keys = R.add_reference_keys(df, table)["reference_keys"].tolist()
        assert keys[0] == "10.1000/aaa; https://openalex.org/W9"
        assert keys[1] == "10.1000/bbb"
        assert keys[2] == "10.1000/ccc; total garbage"

    def test_cross_source_reference_collapses_onto_one_doi(self):
        # An OpenAlex doc references corpus work W1 (which has DOI aaa); a WoS doc
        # references the same work by its DOI. Both must yield the same key.
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1", "https://openalex.org/OA", "WOS:1"],
            "doi": ["10.1000/aaa", "10.1000/oa", "10.1000/wos"],
            "references": [
                None,
                "https://openalex.org/W1",                 # OA doc -> DOI aaa
                "Author, 2010, J, DOI 10.1000/aaa",        # WoS doc -> DOI aaa
            ],
        })
        table = R.seed_table_from_dataframe(df)
        keys = R.add_reference_keys(df, table)["reference_keys"].tolist()
        assert keys[1] == "10.1000/aaa"
        assert keys[2] == "10.1000/aaa"

    def test_empty_references_give_none(self):
        df = pd.DataFrame({"references": [None, "", "   "]})
        assert R.add_reference_keys(df, {})["reference_keys"].tolist() == [None, None, None]


class TestHasUnkeyedReferences:
    """Gate for resolve_references: raw citation strings / DOIs need resolving,
    native id keys do not."""

    def test_raw_wos_strings_are_unkeyed(self):
        df = pd.DataFrame({"references": ["Smith J, 2019, NATURE, V1, P1"]})
        assert R.has_unkeyed_references(df) is True

    def test_bare_doi_is_unkeyed(self):
        df = pd.DataFrame({"references": ["10.1000/abc"]})
        assert R.has_unkeyed_references(df) is True

    def test_pure_openalex_ids_are_keyed(self):
        df = pd.DataFrame({
            "references": ["https://openalex.org/W1; https://openalex.org/W2"],
        })
        assert R.has_unkeyed_references(df) is False

    def test_no_references_at_all(self):
        df = pd.DataFrame({"references": [None, "", "   "]})
        assert R.has_unkeyed_references(df) is False

    def test_missing_column(self):
        assert R.has_unkeyed_references(pd.DataFrame({"id": ["W1"]})) is False

    def test_mixed_sources_are_unkeyed(self):
        df = pd.DataFrame({
            "references": ["https://openalex.org/W1", "Author, 2010, DOI 10.1/z"],
        })
        assert R.has_unkeyed_references(df) is True


class TestCache:

    def test_round_trip_with_none(self, tmp_path):
        path = os.path.join(tmp_path, "cache.csv")
        R.save_cache(path, {"W1": "10.1000/aaa", "W9": None})
        assert R.load_cache(path) == {"W1": "10.1000/aaa", "W9": None}

    def test_missing_file_is_empty(self, tmp_path):
        assert R.load_cache(os.path.join(tmp_path, "nope.csv")) == {}

    def test_none_path_noops(self):
        assert R.load_cache(None) == {}
        R.save_cache(None, {"W1": "x"})  # must not raise


class _FakeOpenAlexClient:
    """Minimal stand-in exposing _fetch_page over a fixed id->doi map."""

    def __init__(self, id_doi):
        self.id_doi = id_doi
        self.calls = 0

    def _fetch_page(self, params):
        self.calls += 1
        # params['filter'] = 'openalex_id:https://openalex.org/W1|...'
        wanted = params["filter"].split(":", 1)[1].split("|")
        results = []
        for url in wanted:
            wid = R._as_openalex_id(url)
            if wid in self.id_doi:
                results.append({"id": f"https://openalex.org/{wid}",
                                "doi": self.id_doi[wid]})
        return {"results": results}


def _oa_resolvers(client):
    """Wire a fake client into the {scheme: ids->{id:doi}} resolver map."""
    return {"openalex": lambda ids: R.resolve_ids_via_openalex(ids, client)}


class TestApiResolution:

    def test_resolve_and_record_missing_as_none(self):
        client = _FakeOpenAlexClient({"W1": "https://doi.org/10.1/aaa"})
        out = R.resolve_ids_via_openalex(["W1", "W2"], client, chunk=50)
        assert out == {"W1": "10.1/aaa", "W2": None}  # W2 not returned -> None

    def test_chunking(self):
        client = _FakeOpenAlexClient({f"W{i}": f"10.1/{i}" for i in range(120)})
        out = R.resolve_ids_via_openalex([f"W{i}" for i in range(120)], client, chunk=50)
        assert client.calls == 3          # 50 + 50 + 20
        assert len(out) == 120


class TestCompleteReferenceKeys:

    def test_end_to_end_seeds_resolves_and_caches(self, tmp_path):
        cache = os.path.join(tmp_path, "wxxxx_doi.csv")
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1", "https://openalex.org/OA"],
            "doi": ["10.1000/aaa", "10.1000/oa"],
            "references": [
                "https://openalex.org/W1; https://openalex.org/W999",  # intra + extra corpus
                "Author, 2010, DOI 10.1000/aaa",
            ],
        })
        # W1 is seeded from the corpus (free); W999 must come from the API.
        client = _FakeOpenAlexClient({"W999": "10.9999/ext"})
        out = R.complete_reference_keys(df, cache_path=cache, resolvers=_oa_resolvers(client))
        keys = out["reference_keys"].tolist()
        assert keys[0] == "10.1000/aaa; 10.9999/ext"
        assert keys[1] == "10.1000/aaa"
        # W999 got written to the persistent cache; W1 came from the seed (free).
        assert R.load_cache(cache)["W999"] == "10.9999/ext"

    def test_pure_openalex_skips_api(self, tmp_path):
        # All references are OpenAlex ids -> nothing to bridge to -> no API call,
        # extra-corpus ids keep their Wxxxx token even with resolve_external on.
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1"],
            "doi": ["10.1000/aaa"],
            "references": ["https://openalex.org/W1; https://openalex.org/W999"],
        })
        client = _FakeOpenAlexClient({"W999": "10.9999/ext"})
        out = R.complete_reference_keys(df, cache_path=None, resolvers=_oa_resolvers(client))
        assert client.calls == 0
        assert out["reference_keys"].tolist() == ["10.1000/aaa; https://openalex.org/W999"]

    def test_other_source_dois_trigger_api(self, tmp_path):
        # A WoS DOI reference is present -> there is something to bridge to -> the
        # API is used to resolve the extra-corpus OpenAlex id.
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1", "WOS:1"],
            "doi": ["10.1000/aaa", "10.1000/wos"],
            "references": [
                "https://openalex.org/W999",
                "Author, 2010, DOI 10.1000/zzz",
            ],
        })
        client = _FakeOpenAlexClient({"W999": "10.9999/ext"})
        out = R.complete_reference_keys(df, cache_path=None, resolvers=_oa_resolvers(client))
        assert client.calls == 1
        assert out["reference_keys"].tolist() == ["10.9999/ext", "10.1000/zzz"]

    def test_offline_keeps_extra_corpus_token(self, tmp_path):
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1"],
            "doi": ["10.1000/aaa"],
            "references": ["https://openalex.org/W999"],
        })
        out = R.complete_reference_keys(df, cache_path=None, resolve_external=False)
        # No resolver -> extra-corpus id keeps its raw token.
        assert out["reference_keys"].tolist() == ["https://openalex.org/W999"]

    def test_no_resolver_for_scheme_keeps_token_even_when_bridgeable(self, tmp_path):
        # A WoS DOI is present (bridgeable) but no OpenAlex resolver is wired in
        # (e.g. no API credentials): the extra-corpus Wxxxx keeps its token, the
        # WoS DOI still maps. Agnostic: a scheme without a resolver is just skipped.
        df = pd.DataFrame({
            "id":  ["https://openalex.org/W1", "WOS:1"],
            "doi": ["10.1000/aaa", "10.1000/wos"],
            "references": ["https://openalex.org/W999", "Author, DOI 10.1000/zzz"],
        })
        out = R.complete_reference_keys(df, cache_path=None, resolvers={})
        assert out["reference_keys"].tolist() == [
            "https://openalex.org/W999", "10.1000/zzz"]
