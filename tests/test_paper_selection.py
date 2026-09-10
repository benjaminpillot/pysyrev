"""
Tests for the reading-list machinery of pysyrev.core.report.sections.papers.

Covers review detection against the real document-type vocabularies of the
sources.

All tests are fast: synthetic DataFrames only, no model and no disk access.

Test naming convention:
  test_<what>_<condition>
"""

import numpy as np
import pandas as pd

from pysyrev.core.report.sections.papers import _is_review


# =============================================================================
# Helpers
# =============================================================================

def _docs(rows):
    """Frame of (document_type, title) pairs."""
    return pd.DataFrame(rows, columns=["document_type", "title"])


# =============================================================================
# _is_review
# =============================================================================

class TestIsReview:

    def test_type_matches_wos_and_openalex_vocabularies(self):
        df = _docs([("Review", "a"), ("review", "b"),
                    ("Review; Early Access", "c"), ("Review; Book Chapter", "d")])
        assert _is_review(df, "type").tolist() == [True] * 4

    def test_type_does_not_match_plain_articles(self):
        df = _docs([("Article", "a"), ("article", "b"),
                    ("Proceedings Paper", "c"), ("preprint", "d")])
        assert _is_review(df, "type").tolist() == [False] * 4

    def test_type_does_not_match_openalex_peer_review(self):
        # OpenAlex types a referee report `peer-review`; a bare substring test
        # would take all of them for literature reviews.
        df = _docs([("peer-review", "Author response: something")])
        assert _is_review(df, "type").tolist() == [False]

    def test_peer_review_excluded_even_when_title_says_review(self):
        # eLife referee reports are titled "Reviewer #2 (Public Review): ..."
        df = _docs([("peer-review", "Reviewer #2 (Public Review): a finding"),
                    ("peer-review", "Joint Public Review: another finding")])
        for mode in ("type", "title", "type_or_title"):
            assert _is_review(df, mode).tolist() == [False, False], mode

    def test_title_matches_untyped_reviews(self):
        # OpenAlex types reviews sparingly; these only declare it in the title.
        df = _docs([("Article", "A systematic review and a way forward"),
                    ("Proceedings Paper", "Review on interdependency modeling")])
        assert _is_review(df, "title").tolist() == [True, True]
        assert _is_review(df, "type").tolist() == [False, False]

    def test_title_rejects_papers_about_peer_review(self):
        # A paper *about* refereeing is not a review of a literature.
        df = _docs([("Article", "The miracle of peer review: an agent-based model"),
                    ("Article", "Evaluating alternative systems of peer-review")])
        assert _is_review(df, "title").tolist() == [False, False]

    def test_title_does_not_match_overview(self):
        df = _docs([("Article", "An overview of energy models")])
        assert _is_review(df, "title").tolist() == [False]

    def test_type_or_title_is_the_union(self):
        df = _docs([("Review", "no keyword here"),
                    ("Article", "A review of methods"),
                    ("Article", "neither")])
        assert _is_review(df, "type_or_title").tolist() == [True, True, False]

    def test_missing_values_are_tolerated(self):
        df = _docs([(None, None), (np.nan, "A review of methods")])
        assert _is_review(df, "type_or_title").tolist() == [False, True]

    def test_empty_frame_returns_empty_boolean_mask(self):
        mask = _is_review(_docs([]).astype(object), "type_or_title")
        assert mask.dtype == bool and len(mask) == 0

    def test_mask_indexes_the_source_frame(self):
        df = _docs([("Article", "x"), ("Review", "y")])
        df.index = [10, 20]
        assert df[_is_review(df, "type")].index.tolist() == [20]
