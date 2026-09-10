"""
Tests for the reading-list machinery of pysyrev.core.report.sections.papers.

Covers the three units the paper-selection section leans on: review detection
against the real document-type vocabularies, the split between the guaranteed
recent-review cohort and the ranked historical one, and the apportionment of a
global reading budget over the topics.

All tests are fast: synthetic DataFrames only, no model and no disk access.

Test naming convention:
  test_<what>_<condition>
"""

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.config import PaperSelectionConfig, ReviewSelectionConfig
from pysyrev.core.report.sections.papers import (
    _allocate_reading_budget,
    _build_paper_selection_section,
    _is_review,
    _pick_reviews,
)


# =============================================================================
# Helpers
# =============================================================================

def _docs(rows):
    """Frame of (document_type, title) pairs."""
    return pd.DataFrame(rows, columns=["document_type", "title"])


def _corpus(spec):
    """A BERTopic-results frame: {topic_id: n_docs}, doc 0 of each topic a
    current-year review, docs 1-2 older reviews, the rest articles.

    The reviews are deliberately the *least* cited papers, so the ranked branch
    would never pick them on its own. That is what makes an additive review
    branch visible as an over-budget table rather than being hidden by the
    dedup that merges a review the ranking had already chosen.
    """
    rows = []
    for topic, n in spec.items():
        for k in range(n):
            is_review = k < 3
            rows.append(dict(
                id=f"t{topic}p{k}", Topic=topic,
                year=2026 if k == 0 else 2010 + k,
                cited_by=(5 - k) if is_review else (1000 - k),
                document_type="Review" if is_review else "Article",
                title=f"T{topic} paper {k}",
                doi=f"10.1/t{topic}p{k}", Document="text",
            ))
    return pd.DataFrame(rows)


def _table_rows(section):
    return next(b for b in section["blocks"] if b["type"] == "table")["rows"]


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


# =============================================================================
# _pick_reviews
# =============================================================================

def _reviews(specs):
    """Frame of (year, cited_by) reviews, titled for readability."""
    return pd.DataFrame([
        dict(year=float(y), cited_by=float(c), title=f"{y}/{c}")
        for y, c in specs
    ])


class TestPickReviews:

    CFG = dict(max_reviews_per_topic=3, relevance_mode="blend",
               recent_years=1, max_recent_reviews=2)

    def _pick(self, specs, current_year=2026.0, **over):
        cfg = ReviewSelectionConfig(**{**self.CFG, **over})
        return _pick_reviews(_reviews(specs), cfg, current_year)["title"].tolist()

    def test_uncited_current_year_review_is_kept(self):
        # The whole point: relevance cannot rank a paper with no time to be
        # cited, so a current-year review must not be ranked against old ones.
        kept = self._pick([(2026, 0), (2012, 400), (2021, 60), (2005, 4)])
        assert "2026/0" in kept

    def test_recent_cohort_leads_the_result(self):
        kept = self._pick([(2012, 400), (2026, 1), (2021, 60)])
        assert kept[0] == "2026/1"

    def test_recent_cohort_capped_newest_first(self):
        kept = self._pick([(2026, 0), (2026, 5), (2026, 9), (2012, 400)],
                          max_recent_reviews=2)
        assert sum(1 for k in kept if k.startswith("2026")) == 2

    def test_historical_cohort_capped(self):
        kept = self._pick([(2012, 400), (2010, 300), (2008, 200), (2005, 4)],
                          max_reviews_per_topic=2, recent_years=0)
        assert len(kept) == 2

    def test_historical_cohort_ranked_by_relevance(self):
        kept = self._pick([(2005, 4), (2012, 400), (2003, 0)],
                          max_reviews_per_topic=2, recent_years=0)
        assert kept[0] == "2012/400"

    def test_dead_old_review_dropped_before_a_cited_one(self):
        kept = self._pick([(2003, 0), (2012, 400), (2021, 60)],
                          max_reviews_per_topic=2, recent_years=0)
        assert "2003/0" not in kept

    def test_recent_years_zero_disables_the_cohort(self):
        # Everything is then ranked, so an uncited current-year review loses.
        kept = self._pick([(2026, 0), (2012, 400), (2021, 60), (2005, 4)],
                          recent_years=0, max_reviews_per_topic=2)
        assert kept == ["2012/400", "2021/60"]

    def test_recent_years_widens_the_window(self):
        kept = self._pick([(2025, 0), (2012, 400)], recent_years=2)
        assert kept[0] == "2025/0"

    # Four reviews, arranged so the two modes disagree on the winner: 2021/60 has
    # the best citations-per-year, 2005/100 the best raw count. Fewer than four
    # and the percentiles tie, leaving the input order to decide.
    SPLIT = [(2021, 60), (2005, 100), (2005, 90), (2015, 20)]

    def test_cpy_ranks_on_the_per_year_rate_alone(self):
        assert self._pick(self.SPLIT, recent_years=0, max_reviews_per_topic=1,
                          relevance_mode="cpy") == ["2021/60"]

    def test_blend_keeps_raw_citations_in_play(self):
        assert self._pick(self.SPLIT, recent_years=0, max_reviews_per_topic=1,
                          relevance_mode="blend") == ["2005/100"]


# =============================================================================
# _allocate_reading_budget
# =============================================================================

class TestAllocateReadingBudget:

    POOLS = [120, 60, 30, 12, 4]

    def test_sqrt_allocation_spends_the_budget(self):
        assert sum(_allocate_reading_budget(self.POOLS, 40, 2)) == 40

    def test_sqrt_allocation_is_monotone_in_pool_size(self):
        quotas = _allocate_reading_budget(self.POOLS, 40, 2)
        assert quotas == sorted(quotas, reverse=True)

    def test_linear_is_steeper_than_sqrt_for_the_largest_topic(self):
        sq = _allocate_reading_budget(self.POOLS, 40, 2, mode="sqrt")
        li = _allocate_reading_budget(self.POOLS, 40, 2, mode="linear")
        assert li[0] > sq[0] and li[-1] <= sq[-1]

    def test_sqrt_lifts_the_smallest_topic_above_the_floor(self):
        quotas = _allocate_reading_budget(self.POOLS, 40, 2, mode="sqrt")
        assert quotas[-1] > 2

    def test_floor_is_honoured_even_past_the_budget(self):
        # A topic missing from the reading list is worse than overspending.
        quotas = _allocate_reading_budget([50] * 20, 10, 2)
        assert quotas == [2] * 20 and sum(quotas) == 40

    def test_ceiling_is_respected(self):
        quotas = _allocate_reading_budget(self.POOLS, 40, 2, max_per_topic=8)
        assert max(quotas) <= 8

    def test_quota_never_exceeds_the_pool(self):
        quotas = _allocate_reading_budget([100, 3, 2], 40, 2)
        assert quotas[1] <= 3 and quotas[2] <= 2

    def test_budget_larger_than_the_corpus_takes_everything(self):
        assert _allocate_reading_budget([5, 4, 3], 100, 2) == [5, 4, 3]

    def test_empty_pool_gets_nothing(self):
        assert _allocate_reading_budget([50, 0, 10], 20, 2)[1] == 0

    def test_single_topic_takes_the_whole_budget(self):
        assert _allocate_reading_budget([50], 20, 2) == [20]

    def test_zero_floor_is_allowed(self):
        quotas = _allocate_reading_budget([100, 1], 10, 0)
        assert sum(quotas) == 10 and all(q >= 0 for q in quotas)


# =============================================================================
# Budget, end to end through the section builder
# =============================================================================

class TestBudgetInSection:

    POOLS = {0: 120, 1: 60, 2: 30, 3: 12, 4: 4}

    def _rows(self, **over):
        cfg = PaperSelectionConfig(min_year=2000, **over)
        section = _build_paper_selection_section(
            _corpus(self.POOLS), None, {}, cfg, None, 1)
        return _table_rows(section)

    def test_budget_is_the_number_of_papers_listed(self):
        assert len(self._rows(budget=40)) == 40

    def test_reviews_spend_the_budget_rather_than_adding_to_it(self):
        with_reviews = len(self._rows(budget=40))
        without = len(self._rows(budget=40, review=dict(enabled=False)))
        assert with_reviews == without == 40

    def test_bigger_topic_gets_a_bigger_share(self):
        rows = self._rows(budget=40)
        per_topic = pd.Series([r[0] for r in rows]).value_counts()
        assert per_topic["0"] > per_topic["4"]

    def test_every_topic_appears(self):
        rows = self._rows(budget=40)
        assert {r[0] for r in rows} == {str(t) for t in self.POOLS}

    def test_proportion_applies_when_no_budget_is_set(self):
        # Legacy path: a share of each topic's own corpus, reviews on top.
        rows = self._rows(proportion_per_topic=0.15)
        assert len(rows) != 40
