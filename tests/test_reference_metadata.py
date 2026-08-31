"""Tests for reference metadata — co-citation nodes resolved back to real works.

A co-citation node is a reference, usually outside the corpus, so nothing local
knows its title, year or authors. `fetch_reference_metadata` routes each key to
an id scheme, hands the missing ids to that scheme's fetcher, and caches the
answer (negatives included) so a rerun costs nothing. `build_cocitation` then
turns that metadata into per-community TF-IDF terms, and `cluster_profiles` into
the authors/venues/period profile of each community.

The provider is a plug-in: these tests use a fake fetcher for the generic layer
and a fake OpenAlex client for the OpenAlex one, so nothing hits the network.
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from pysyrev.core import references as R
from pysyrev.core.networks import (
    build_cocitation,
    cluster_profiles,
    frequent_references,
    reference_texts,
)


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------

def make_meta(**over):
    meta = {"title": "A title", "doi": "10.1000/a", "year": 2010,
            "authors": ["Ann Author"], "venue": "J Test", "cited_by_count": 7,
            "abstract": None}
    meta.update(over)
    return meta


class RecordingFetcher:
    """Fetcher double: answers for the ids it knows, records every batch."""

    def __init__(self, known):
        self.known = known
        self.calls = []

    def __call__(self, ids):
        ids = list(ids)
        self.calls.append(ids)
        return {i: self.known[i] for i in ids if i in self.known}


class FakeOpenAlexClient:
    """Stands in for OpenAlexClient: serves works `W<n>` with DOI `10.1000/w<n>`."""

    def __init__(self, known_ids):
        self.known_ids = {w.upper() for w in known_ids}
        self.pages = []

    def _fetch_page(self, params):
        self.pages.append(params)
        scheme, values = params["filter"].split(":", 1)
        results = []
        for value in values.split("|"):
            work = (value.rsplit("/", 1)[-1].upper() if scheme == "openalex_id"
                    else "W" + value.rsplit("/w", 1)[-1])
            if work.upper() not in self.known_ids:
                continue
            results.append({
                "id": f"https://openalex.org/{work}",
                "doi": f"https://doi.org/10.1000/{work.lower()}",
                "display_name": f"Work {work}",
                "publication_year": 2000,
                "cited_by_count": 3,
                "authorships": [{"author": {"display_name": "Ann Author"}}],
                "primary_location": {"source": {"display_name": "J Test"}},
            })
        return {"results": results}


@pytest.fixture
def two_block_corpus():
    """40 documents citing within one of two disjoint reference pools — a corpus
    whose co-citation network has two obvious communities."""
    rng = np.random.default_rng(0)
    pool_a = [f"W{i}" for i in range(1, 16)]
    pool_b = [f"W{i}" for i in range(100, 115)]
    rows = [{"id": f"D{d}",
             "references": "; ".join(rng.choice(pool_a if d % 2 == 0 else pool_b,
                                                6, replace=False))}
            for d in range(40)]
    return pd.DataFrame(rows), pool_a, pool_b


# ---------------------------------------------------------------------------
# Key routing
# ---------------------------------------------------------------------------

class TestKeyScheme:

    @pytest.mark.parametrize("key, expected", [
        ("W123",                          ("openalex", "W123")),
        ("https://openalex.org/W123",     ("openalex", "W123")),
        ("10.1000/ABC",                   ("doi", "10.1000/abc")),
        ("https://doi.org/10.1000/abc",   ("doi", "10.1000/abc")),
        ("Smith J, 2019, NATURE, DOI 10.1000/abc", ("doi", "10.1000/abc")),
        ("Smith J, 2019, NATURE, V1, P2", None),
        ("", None),
        (None, None),
    ])
    def test_routes_every_key_form(self, key, expected):
        assert R._key_scheme(key) == expected


# ---------------------------------------------------------------------------
# Generic fetch + cache
# ---------------------------------------------------------------------------

class TestFetchReferenceMetadata:

    def test_returns_metadata_keyed_by_original_key(self):
        fetcher = RecordingFetcher({"W1": make_meta(title="One")})
        meta = R.fetch_reference_metadata(
            ["https://openalex.org/W1"], {"openalex": fetcher})
        # keyed by what the caller passed, not by the canonical form
        assert list(meta) == ["https://openalex.org/W1"]
        assert meta["https://openalex.org/W1"]["title"] == "One"

    def test_routes_each_key_to_its_own_scheme(self):
        works = RecordingFetcher({"W1": make_meta(title="Work")})
        dois = RecordingFetcher({"10.1000/a": make_meta(title="Doi")})
        meta = R.fetch_reference_metadata(
            ["W1", "10.1000/A"], {"openalex": works, "doi": dois})
        assert works.calls == [["W1"]]
        assert dois.calls == [["10.1000/a"]]
        assert meta["W1"]["title"] == "Work"
        assert meta["10.1000/A"]["title"] == "Doi"

    def test_skips_keys_without_a_queryable_identity(self):
        fetcher = RecordingFetcher({})
        meta = R.fetch_reference_metadata(
            ["Smith J, 2019, NATURE, V1, P2"], {"openalex": fetcher, "doi": fetcher})
        assert meta == {}
        assert fetcher.calls == []          # nothing to ask about

    def test_skips_schemes_without_a_fetcher(self):
        dois = RecordingFetcher({"10.1000/a": make_meta()})
        meta = R.fetch_reference_metadata(["W1", "10.1000/a"], {"doi": dois})
        assert list(meta) == ["10.1000/a"]     # the OpenAlex id is left unresolved

    def test_unresolved_id_yields_an_empty_entry(self):
        fetcher = RecordingFetcher({})
        meta = R.fetch_reference_metadata(["W1"], {"openalex": fetcher})
        assert meta["W1"] == {"title": None, "doi": None, "year": None,
                              "authors": [], "venue": None,
                              "cited_by_count": None, "abstract": None}

    def test_cache_round_trip_avoids_a_second_fetch(self, tmp_path):
        cache = str(tmp_path / "meta.csv")
        fetcher = RecordingFetcher({"W1": make_meta(title="One", authors=["A", "B"])})
        first = R.fetch_reference_metadata(["W1"], {"openalex": fetcher},
                                           cache_path=cache)
        second = R.fetch_reference_metadata(["W1"], {"openalex": fetcher},
                                            cache_path=cache)
        assert len(fetcher.calls) == 1      # the second run never asked
        assert second == first
        assert second["W1"]["authors"] == ["A", "B"]

    def test_cache_remembers_the_negative(self, tmp_path):
        cache = str(tmp_path / "meta.csv")
        fetcher = RecordingFetcher({})
        R.fetch_reference_metadata(["W1"], {"openalex": fetcher}, cache_path=cache)
        R.fetch_reference_metadata(["W1"], {"openalex": fetcher}, cache_path=cache)
        assert len(fetcher.calls) == 1      # "no record" is cached too

    def test_only_the_missing_ids_are_fetched(self, tmp_path):
        cache = str(tmp_path / "meta.csv")
        fetcher = RecordingFetcher({"W1": make_meta(), "W2": make_meta()})
        R.fetch_reference_metadata(["W1"], {"openalex": fetcher}, cache_path=cache)
        R.fetch_reference_metadata(["W1", "W2"], {"openalex": fetcher},
                                   cache_path=cache)
        assert fetcher.calls == [["W1"], ["W2"]]

    def test_missing_cache_file_is_not_an_error(self, tmp_path):
        assert R.load_metadata_cache(str(tmp_path / "absent.csv")) == {}
        assert R.load_metadata_cache(None) == {}

    def test_cache_directory_is_created(self, tmp_path):
        # a cache path is configuration, typically relative ('cache/meta.csv'):
        # its directory should not have to be pre-created by hand
        cache = str(tmp_path / "cache" / "reference_metadata.csv")
        fetcher = RecordingFetcher({"W1": make_meta()})
        R.fetch_reference_metadata(["W1"], {"openalex": fetcher}, cache_path=cache)
        assert R.load_metadata_cache(cache)["W1"]["title"] == "A title"

    def test_unwritable_cache_does_not_discard_the_fetch(self, tmp_path, capsys):
        # the API calls have already been paid for: a cache that cannot be
        # written costs the *next* run, not this one
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        try:
            fetcher = RecordingFetcher({"W1": make_meta(title="Kept")})
            meta = R.fetch_reference_metadata(
                ["W1"], {"openalex": fetcher},
                cache_path=str(blocked / "sub" / "meta.csv"))
            assert meta["W1"]["title"] == "Kept"
            assert "cache not written" in capsys.readouterr().out
        finally:
            blocked.chmod(0o700)


# ---------------------------------------------------------------------------
# OpenAlex fetcher
# ---------------------------------------------------------------------------

class TestOpenAlexFetcher:

    def test_serves_both_id_and_doi_schemes(self):
        client = FakeOpenAlexClient(["W1", "W2"])
        fetch = partial(R.fetch_metadata_via_openalex, client=client,
                        show_progress=False)
        out = fetch(["W1", "10.1000/w2"])
        assert out["W1"]["title"] == "Work W1"
        assert out["W1"]["doi"] == "10.1000/w1"
        assert out["10.1000/w2"]["title"] == "Work W2"
        # one request per scheme, not one per id
        assert len(client.pages) == 2
        assert {p["filter"].split(":", 1)[0] for p in client.pages} == {
            "openalex_id", "doi"}

    def test_unknown_ids_are_absent(self):
        client = FakeOpenAlexClient(["W1"])
        out = R.fetch_metadata_via_openalex(["W1", "W9"], client=client,
                                            show_progress=False)
        assert set(out) == {"W1"}

    def test_batches_respect_the_chunk_size(self):
        ids = [f"W{i}" for i in range(1, 8)]
        client = FakeOpenAlexClient(ids)
        R.fetch_metadata_via_openalex(ids, client=client, chunk=3,
                                      show_progress=False)
        assert [len(p["filter"].split("|")) for p in client.pages] == [3, 3, 1]

    def test_abstracts_are_opt_in(self):
        client = FakeOpenAlexClient(["W1"])
        R.fetch_metadata_via_openalex(["W1"], client=client, show_progress=False)
        assert "abstract_inverted_index" not in client.pages[0]["select"]
        R.fetch_metadata_via_openalex(["W1"], client=client, show_progress=False,
                                      include_abstracts=True)
        assert "abstract_inverted_index" in client.pages[1]["select"]

    def test_abstract_is_rebuilt_from_the_inverted_index(self):
        record = {"abstract_inverted_index": {"agent": [0, 3], "based": [1],
                                              "models": [2]}}
        assert R._openalex_abstract(record) == "agent based models agent"

    def test_authors_fall_back_to_the_raw_name(self):
        record = {"authorships": [{"author": {}, "raw_author_name": "Ann A"},
                                  {"author": {"display_name": "Bob B"}}]}
        assert R._openalex_authors(record) == ["Ann A", "Bob B"]


# ---------------------------------------------------------------------------
# Consumption by the co-citation network
# ---------------------------------------------------------------------------

class TestCocitationWithMetadata:

    def test_frequent_references_matches_the_matrix_nodes(self, two_block_corpus):
        df, _, _ = two_block_corpus
        from pysyrev.core.networks import cocitation_matrix, reference_sets
        _, refsets = reference_sets(df, ref_col="references")
        kept, _ = frequent_references(refsets, min_ref_freq=2)
        node_ids, _ = cocitation_matrix(refsets, min_ref_freq=2)
        assert kept == node_ids             # the node set can be derived up front

    def test_reference_texts_are_row_aligned(self):
        meta = {"W1": make_meta(title="T", venue="V", abstract="A")}
        assert reference_texts(["W1", "W2"], meta) == ["T. V. A", ""]

    def test_metadata_fills_terms_and_node_meta(self, two_block_corpus):
        df, pool_a, pool_b = two_block_corpus
        meta = {r: make_meta(
                    title=("agent based simulation of energy transition policy"
                           if r in pool_a else
                           "travel demand mobility choice urban commuting"),
                    venue="J Energy" if r in pool_a else "J Transport")
                for r in pool_a + pool_b}
        res = build_cocitation(df, ref_col="references", min_ref_freq=2,
                               min_size=3, ref_meta=meta)
        assert res.n_communities == 2
        assert set(res.terms) == {0, 1}
        # each community's terms come from its own references' titles
        joined = {c: " ".join(t) for c, t in res.terms.items()}
        assert any("energy" in t for t in joined.values())
        assert any("mobility" in t or "commuting" in t for t in joined.values())
        assert len(res.node_meta) == res.n_nodes

    def test_build_without_metadata_is_unchanged(self, two_block_corpus):
        df, _, _ = two_block_corpus
        res = build_cocitation(df, ref_col="references", min_ref_freq=2,
                               min_size=3)
        assert res.terms == {}
        assert res.node_meta == {}
        assert res.n_communities == 2       # the network itself is unaffected

    def test_cluster_profiles_summarise_authors_venues_and_period(
            self, two_block_corpus):
        df, pool_a, pool_b = two_block_corpus
        meta = {r: make_meta(title=f"Title {r}",
                             year=1990 if r in pool_a else 2015,
                             authors=["Ann Author"] if r in pool_a else ["Bob Writer"],
                             venue="J Energy" if r in pool_a else "J Transport")
                for r in pool_a + pool_b}
        res = build_cocitation(df, ref_col="references", min_ref_freq=2,
                               min_size=3, ref_meta=meta)
        profiles = cluster_profiles(res.node_ids, res.labels, res.W, res.node_meta)

        assert set(profiles) == {0, 1}
        by_author = {p["authors"][0][0]: p for p in profiles.values()}
        assert set(by_author) == {"Ann Author", "Bob Writer"}
        assert by_author["Ann Author"]["median_year"] == 1990
        assert by_author["Bob Writer"]["venues"][0][0] == "J Transport"
        for profile in profiles.values():
            assert profile["n_known"] == profile["n_refs"]
            # works are ordered by co-citation strength
            strengths = [w["strength"] for w in profile["works"]]
            assert strengths == sorted(strengths, reverse=True)

    def test_cluster_profiles_empty_without_metadata(self, two_block_corpus):
        df, _, _ = two_block_corpus
        res = build_cocitation(df, ref_col="references", min_ref_freq=2,
                               min_size=3)
        assert cluster_profiles(res.node_ids, res.labels, res.W, {}) == {}
