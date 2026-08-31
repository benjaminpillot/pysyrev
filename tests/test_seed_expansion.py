"""Tests for seed-driven corpus expansion (seeds -> candidate pool)."""

import pandas as pd
import pytest

from pysyrev.bibdata import OpenAlexDataset
from pysyrev.core.bib import DEFAULT_FIELDS, check_bib_dataset
from pysyrev.core.config import OpenAlexSeedConfig, OpenAlexSourceConfig
from pysyrev.core.seed_expansion import OpenAlexExpander, dedup_pool, read_seed_file
from pysyrev.core.seed_expansion.base import normalize_doi, title_key


# =============================================================================
# Fake OpenAlex client
# =============================================================================

def _work(wid, title="A title", doi=None, refs=(), year=2020, abstract="Hello world"):
    """A minimally realistic OpenAlex work JSON."""
    return {
        "id": f"https://openalex.org/{wid}",
        "title": title,
        "publication_year": year,
        "cited_by_count": 3,
        "language": "en",
        "type": "article",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "primary_location": {"source": {"display_name": "Energy Policy"}},
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "keywords": [],
        "abstract_inverted_index": ({w: [i] for i, w in enumerate(abstract.split())}
                                    if abstract else None),
        "authorships": [{"author": {"display_name": "Jane Doe"},
                         "raw_affiliation_strings": ["Univ X"]}],
    }


class _FakeSearchResult:
    def __init__(self, records):
        self.records = records
        self.total_found = len(records)


class _FakeClient:
    """Stands in for OpenAlexClient: answers id batches and filter searches
    from a fixed catalogue, and records every request it served."""

    def __init__(self, works, cites=None, query_hits=None):
        self.works = {w["id"]: w for w in works}         # by full URL id
        self.by_doi = {normalize_doi(w["doi"]): w for w in works if w["doi"]}
        self.cites = cites or {}                          # {Wxxxx: [work, ...]}
        self.query_hits = query_hits or {}                # {query: [work, ...]}
        self.id_requests = []
        self.searches = []

    # -- batched id lookup (expander calls the private page primitive) --
    def _fetch_page(self, params):
        self.id_requests.append(params["filter"])
        key, _, values = params["filter"].partition(":")
        results = []
        for value in values.split("|"):
            if key == "openalex_id":
                hit = self.works.get(f"https://openalex.org/{value}")
            else:
                hit = self.by_doi.get(value)
            if hit is not None:
                results.append(hit)
        return {"results": results}

    # -- paginated filter search --
    def search(self, query=None, filters=None, max_records=None, show_progress=True):
        self.searches.append(filters)
        records = []
        if "cites" in filters:
            records = self.cites.get(filters["cites"], [])
        elif "title_and_abstract.search" in filters:
            records = self.query_hits.get(filters["title_and_abstract.search"], [])
        return _FakeSearchResult(records[:max_records] if max_records else records)


# =============================================================================
# Seed input
# =============================================================================

class TestSeedFile:

    def _write(self, tmp_path, text):
        path = tmp_path / "seeds.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_one_doi_per_line(self, tmp_path):
        path = self._write(tmp_path, "10.1000/aaa\n10.1000/bbb\n")
        assert read_seed_file(path) == ["10.1000/aaa", "10.1000/bbb"]

    def test_skips_blanks_and_comments(self, tmp_path):
        path = self._write(
            tmp_path,
            "# my seeds\n\n10.1000/aaa\n\n   # another comment\n  10.1000/bbb  \n")
        assert read_seed_file(path) == ["10.1000/aaa", "10.1000/bbb"]

    def test_keeps_first_token_only(self, tmp_path):
        path = self._write(tmp_path, "10.1000/aaa\tSmith et al. 2020\n")
        assert read_seed_file(path) == ["10.1000/aaa"]

    def test_accepts_urls_and_openalex_ids(self, tmp_path):
        path = self._write(tmp_path, "https://doi.org/10.1000/aaa\nW42\n")
        assert read_seed_file(path) == ["https://doi.org/10.1000/aaa", "W42"]


class TestIdentifiers:

    @pytest.mark.parametrize("token, expected", [
        ("10.1000/AAA", "10.1000/aaa"),
        ("https://doi.org/10.1000/aaa", "10.1000/aaa"),
        ("http://dx.doi.org/10.1000/aaa", "10.1000/aaa"),
        ("  10.1000/aaa  ", "10.1000/aaa"),
        ("W42", None),
        ("not an id", None),
        (None, None),
    ])
    def test_normalize_doi(self, token, expected):
        assert normalize_doi(token) == expected

    @pytest.mark.parametrize("token, expected", [
        ("10.1000/aaa", "doi:10.1000/aaa"),
        ("https://doi.org/10.1000/AAA", "doi:10.1000/aaa"),
        ("W42", "W42"),
        ("w42", "W42"),
        ("https://openalex.org/W42", "W42"),
        ("garbage", None),
    ])
    def test_normalize_seed(self, token, expected):
        assert OpenAlexExpander(_FakeClient([])).normalize_seed(token) == expected


# =============================================================================
# Deduplication
# =============================================================================

class TestDedupPool:

    def _pool(self, rows):
        return pd.DataFrame(rows, columns=list(DEFAULT_FIELDS.keys()))

    def test_drops_duplicate_ids(self):
        pool = self._pool([{"id": "W1", "title": "A"}, {"id": "W1", "title": "A"}])
        assert list(dedup_pool(pool).id) == ["W1"]

    def test_collapses_same_title_keeping_the_richest_copy(self):
        pool = self._pool([
            {"id": "W1", "title": "Energy justice, revisited!", "doi": None,
             "abstract": "short"},
            {"id": "W2", "title": "energy justice revisited", "doi": "10.1/aaa",
             "abstract": "a much longer abstract"},
        ])
        deduped = dedup_pool(pool)
        assert list(deduped.id) == ["W2"]

    def test_untitled_records_are_kept_apart(self):
        pool = self._pool([{"id": "W1", "title": None}, {"id": "W2", "title": None}])
        assert list(dedup_pool(pool).id) == ["W1", "W2"]

    def test_preserves_pool_order(self):
        pool = self._pool([{"id": f"W{i}", "title": f"T{i}"} for i in range(4)])
        assert list(dedup_pool(pool).id) == ["W0", "W1", "W2", "W3"]

    def test_title_key_ignores_punctuation_and_case(self):
        assert title_key("Energy Justice: a Review!") == title_key("energy justice a review")


# =============================================================================
# Expansion arms
# =============================================================================

class TestExpansion:

    def _setup(self):
        seed = _work("W1", "Seed paper", doi="10.1000/aaa", refs=["W2"])
        cited = _work("W2", "Cited by the seed")
        citing = _work("W3", "Cites the seed")
        matched = _work("W4", "Found by a text query")
        client = _FakeClient([seed, cited, citing, matched],
                             cites={"W1": [citing]},
                             query_hits={"energy justice": [matched]})
        return client, OpenAlexExpander(client, show_progress=False)

    def test_unions_the_four_arms(self):
        client, expander = self._setup()
        result = expander.expand(["10.1000/aaa"], queries=["energy justice"])

        assert set(result.dataset.id) == {f"https://openalex.org/W{i}" for i in (1, 2, 3, 4)}
        assert result.counts == {"seeds": 1, "forward": 1, "backward": 1,
                                 "queries": 1, "raw_union": 4, "deduped": 4}
        assert result.seed_ids == ["https://openalex.org/W1"]
        assert result.unresolved_seeds == []

    def test_citation_arms_can_be_switched_off(self):
        client, expander = self._setup()
        result = expander.expand(["W1"], queries=["energy justice"],
                                 use_forward=False, use_backward=False)

        assert set(result.dataset.id) == {"https://openalex.org/W1",
                                          "https://openalex.org/W4"}
        assert "forward" not in result.counts and "backward" not in result.counts

    def test_pool_conforms_to_the_bib_schema(self):
        client, expander = self._setup()
        result = expander.expand(["10.1000/aaa"])
        check_bib_dataset(result.dataset)   # raises if a column is missing

    def test_year_window_is_passed_to_the_provider(self):
        client, expander = self._setup()
        expander.expand(["W1"], queries=["energy justice"], year_min=2015, year_max=2024)

        for filters in client.searches:
            assert filters["from_publication_date"] == "2015-01-01"
            assert filters["to_publication_date"] == "2024-12-31"

    def test_unknown_seeds_are_reported_not_fatal(self):
        client, expander = self._setup()
        result = expander.expand(["10.1000/aaa", "10.9999/unknown", "garbage"])

        assert set(result.unresolved_seeds) == {"doi:10.9999/unknown", "garbage"}
        assert result.counts["seeds"] == 1

    def test_no_valid_seed_raises(self):
        client, expander = self._setup()
        with pytest.raises(ValueError, match="No usable seed"):
            expander.expand(["garbage", ""])

    def test_empty_seed_list_says_so(self):
        client, expander = self._setup()
        with pytest.raises(ValueError, match="no seed to start from"):
            expander.expand([])

    def test_seed_ids_are_batched_into_one_request(self):
        seeds = [_work(f"W{i}", f"Paper {i}") for i in range(60)]
        client = _FakeClient(seeds)
        expander = OpenAlexExpander(client, show_progress=False)
        expander.expand([f"W{i}" for i in range(60)],
                        use_forward=False, use_backward=False)

        # 60 ids -> 2 batched id requests (chunk of 50), not 60 single lookups.
        assert len(client.id_requests) == 2


# =============================================================================
# Config wiring
# =============================================================================

class TestSeedConfig:

    def test_source_seed_parses_the_seed_block(self, tmp_path):
        (tmp_path / "seeds.txt").write_text("10.1000/aaa\n", encoding="utf-8")
        config = OpenAlexSourceConfig(source="seed", seed={
            "api_key": "k", "email": "a@b.c", "file": str(tmp_path / "seeds.txt"),
            "queries": ["energy justice"], "year_min": 2010,
        })
        assert isinstance(config.seed, OpenAlexSeedConfig)
        assert config.seed.seed_tokens() == ["10.1000/aaa"]
        assert config.credentials is config.seed      # reused by the ref-key resolver

    def test_pre_rename_keys_still_parse(self, tmp_path):
        # `phrases` / `max_per_phrase` were renamed once it became clear the
        # entries are AND-ed queries, not phrase matches; configs written
        # against the old spelling must keep working.
        (tmp_path / "seeds.txt").write_text("10.1000/aaa\n", encoding="utf-8")
        config = OpenAlexSourceConfig(source="seed", seed={
            "api_key": "k", "file": str(tmp_path / "seeds.txt"),
            "phrases": ["energy justice"], "max_per_phrase": 42,
        })
        assert config.seed.queries == ["energy justice"]
        assert config.seed.max_per_query == 42

    def test_credentials_follow_the_active_source(self):
        # A config file commonly keeps an unused block next to the active one;
        # only the active block is parsed, so credentials must key on `source`.
        config = OpenAlexSourceConfig(
            source="api",
            api={"api_key": "k"},
            seed={"api_key": "other", "seeds": ["10.1000/aaa"]},
        )
        assert config.credentials is config.api
        assert isinstance(config.seed, dict)          # left untouched

    def test_seed_block_is_required(self):
        with pytest.raises(ValueError, match="no `seed:` block"):
            OpenAlexSourceConfig(source="seed")

    def test_seeds_must_come_from_a_file_or_an_inline_list(self):
        with pytest.raises(ValueError, match="one DOI per line"):
            OpenAlexSeedConfig(api_key="k")

    def test_file_and_inline_seeds_are_merged_and_deduplicated(self, tmp_path):
        (tmp_path / "seeds.txt").write_text("10.1000/aaa\n10.1000/bbb\n", encoding="utf-8")
        config = OpenAlexSeedConfig(api_key="k", file=str(tmp_path / "seeds.txt"),
                                    seeds=["10.1000/bbb", "W42"])
        assert config.seed_tokens() == ["10.1000/aaa", "10.1000/bbb", "W42"]

    def test_dataset_is_built_from_the_seed_source(self, tmp_path, monkeypatch):
        (tmp_path / "seeds.txt").write_text("10.1000/aaa\n", encoding="utf-8")
        seed = _work("W1", "Seed paper", doi="10.1000/aaa", refs=["W2"])
        client = _FakeClient([seed, _work("W2", "Cited by the seed")])
        monkeypatch.setattr(OpenAlexDataset, "_client", staticmethod(lambda cfg: client))

        config = OpenAlexSourceConfig(source="seed", seed={
            "api_key": "k", "file": str(tmp_path / "seeds.txt"), "use_forward": False,
        })
        dataset = OpenAlexDataset.from_config(config).dataset

        assert set(dataset.title) == {"Seed paper", "Cited by the seed"}
        check_bib_dataset(dataset)
