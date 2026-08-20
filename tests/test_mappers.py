"""Tests for API-response -> DEFAULT_FIELDS mappers."""

import types

import pandas as pd
import pytest

from pysyrev.core import mappers as M
from pysyrev.core.bib import DEFAULT_FIELDS, check_bib_dataset, _is_string_like


def _oa_record(**overrides):
    """A minimally realistic OpenAlex work JSON, overridable per test."""
    record = {
        "id": "https://openalex.org/W1",
        "title": "Justice in energy transition models",
        "publication_year": 2021,
        "cited_by_count": 7,
        "language": "en",
        "type": "article",
        "doi": "https://doi.org/10.1000/aaa",
        "primary_location": {"source": {"display_name": "Energy Policy"}},
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "keywords": [{"display_name": "energy justice"}],
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [
            {
                "author": {"display_name": "Jane Doe"},
                "raw_affiliation_strings": ["Dept X, Univ Y"],
            },
            {
                "author": {"display_name": "John Roe"},
                "raw_affiliation_strings": ["Inst Z"],
            },
        ],
    }
    record.update(overrides)
    return record


class TestMapOpenAlexRecord:

    def test_authorships_are_joined(self):
        row = M._map_openalex_record(_oa_record())
        assert row["author"] == "Jane Doe; John Roe"
        assert row["affiliations"] == "Dept X, Univ Y; Inst Z"

    def test_core_fields(self):
        row = M._map_openalex_record(_oa_record())
        assert row["id"] == "https://openalex.org/W1"
        assert row["journal"] == "Energy Policy"
        assert row["references"] == "https://openalex.org/W2; https://openalex.org/W3"
        assert row["abstract"] == "Hello world"

    def test_missing_authorships_give_none(self):
        row = M._map_openalex_record(_oa_record(authorships=[]))
        assert row["author"] is None
        assert row["affiliations"] is None

    def test_author_without_display_name_is_skipped(self):
        record = _oa_record(authorships=[
            {"author": {}, "raw_affiliation_strings": ["Inst Z"]},
            {"author": {"display_name": "Jane Doe"}, "raw_affiliation_strings": []},
        ])
        row = M._map_openalex_record(record)
        assert row["author"] == "Jane Doe"
        assert row["affiliations"] == "Inst Z"


class TestFromOpenAlexResult:

    def test_all_columns_present_and_typed(self):
        result = types.SimpleNamespace(records=[_oa_record()])
        df = M.from_openalex_result(result)
        assert list(df.columns) == list(DEFAULT_FIELDS.keys())
        # The regression that broke the pipeline: all-empty text columns were
        # inferred as float64 and rejected by check_bib_dataset. author/affiliations
        # must now be string-like.
        assert _is_string_like(df["author"])
        assert _is_string_like(df["affiliations"])
        check_bib_dataset(df)  # must not raise

    def test_empty_authorships_still_string_typed(self):
        # Even when no record carries authors, the columns must stay string-like
        # (object with None) rather than float64.
        result = types.SimpleNamespace(records=[_oa_record(authorships=[])])
        df = M.from_openalex_result(result)
        assert _is_string_like(df["author"])
        assert _is_string_like(df["affiliations"])
        check_bib_dataset(df)
