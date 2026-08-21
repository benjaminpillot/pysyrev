"""Tests for the three-axis paper ranking, focused on the current-year split."""

import numpy as np

from pysyrev.core.paper_ranking import top_papers_3axis


def _records(years, cites):
    return [{"publication_year": y, "cited_by_count": c,
             "_text": f"energy justice model {i}"}
            for i, (y, c) in enumerate(zip(years, cites))]


def _W(n):
    return np.ones((n, n)) - np.eye(n)


class TestSplitCurrentYear:

    def test_split_returns_historical_and_fronts(self):
        recs = _records([2020, 2021, 2022, 2026], [50, 30, 10, 2])
        out = top_papers_3axis(recs, [0, 0, 0, 0], _W(4), n=10, current_year=2026,
                               drop_current_year=True, split_current_year=True,
                               text_of=lambda r: r["_text"])
        assert set(out) == {"historical", "fronts"}
        hist_years = [recs[r["index"]]["publication_year"] for r in out["historical"][0]]
        front_years = [recs[r["index"]]["publication_year"] for r in out["fronts"][0]]
        assert sorted(hist_years) == [2020, 2021, 2022]
        assert front_years == [2026]

    def test_fronts_are_two_axis(self):
        recs = _records([2020, 2026], [50, 5])
        out = top_papers_3axis(recs, [0, 0], _W(2), n=10, current_year=2026,
                               drop_current_year=True, split_current_year=True,
                               text_of=lambda r: r["_text"])
        front = out["fronts"][0][0]
        assert front["relevance_dropped"] is True
        assert front["relevance"] is None                 # relevance axis dropped
        hist = out["historical"][0][0]
        assert hist["relevance_dropped"] is False
        assert hist["relevance"] is not None              # scored on all three axes

    def test_no_split_mixes_cohorts(self):
        recs = _records([2020, 2026], [50, 5])
        out = top_papers_3axis(recs, [0, 0], _W(2), n=10, current_year=2026,
                               split_current_year=False, text_of=lambda r: r["_text"])
        assert set(out) == {0}                             # plain {cluster: rows}
        assert len(out[0]) == 2                            # both cohorts in one list

    def test_per_cohort_truncation(self):
        # 3 historical + 2 fronts, n=1 → one of each, not one overall.
        recs = _records([2018, 2019, 2020, 2026, 2026], [90, 60, 30, 4, 2])
        out = top_papers_3axis(recs, [0] * 5, _W(5), n=1, current_year=2026,
                               drop_current_year=True, split_current_year=True,
                               text_of=lambda r: r["_text"])
        assert len(out["historical"][0]) == 1
        assert len(out["fronts"][0]) == 1
