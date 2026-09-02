"""Tests for the abstract length cap.

Sources put scraped pages and full texts in the abstract field; those records
carry most of the corpus' characters and most of the review bill. The cap cuts
them back, and must do so without mangling real abstracts.
"""

import pandas as pd
import pytest

from pysyrev.core.clean import truncate_abstract, truncate_abstracts


class TestTruncateAbstract:

    def test_short_text_is_untouched(self):
        text = "A short but complete abstract. It has two sentences."
        assert truncate_abstract(text, 2500) is text

    def test_text_at_the_cap_is_untouched(self):
        text = 'x' * 2500
        assert truncate_abstract(text, 2500) is text

    def test_long_text_is_cut_to_at_most_the_cap(self):
        assert len(truncate_abstract('x' * 10_000, 2500)) <= 2500

    def test_the_cut_lands_on_a_sentence_boundary(self):
        text = ('Sentence one is here. ' * 200)
        out = truncate_abstract(text, 2500)
        assert out.endswith('.')
        assert len(out) <= 2500

    def test_a_sentence_boundary_is_ignored_when_it_costs_too_much(self):
        """A single early period must not shrink the text to nothing."""
        text = 'Intro. ' + 'x' * 10_000
        out = truncate_abstract(text, 2500)
        assert len(out) > 2500 * 0.75

    def test_text_without_punctuation_is_still_cut(self):
        """The reference dumps have no sentence structure at all."""
        text = 'Google Scholar J ' * 2000
        out = truncate_abstract(text, 2500)
        assert len(out) <= 2500

    def test_non_string_values_pass_through(self):
        assert truncate_abstract(None, 2500) is None
        assert pd.isna(truncate_abstract(float('nan'), 2500))

    def test_no_cap_is_a_no_op(self):
        text = 'x' * 10_000
        assert truncate_abstract(text, None) is text


class TestTruncateAbstracts:

    def test_series_is_capped_elementwise(self):
        series = pd.Series(['x' * 100, 'y' * 10_000, None])
        out = truncate_abstracts(series, 2500)
        assert out.iloc[0] == 'x' * 100
        assert len(out.iloc[1]) <= 2500
        assert pd.isna(out.iloc[2])

    def test_falsy_cap_returns_the_series_unchanged(self):
        series = pd.Series(['x' * 10_000])
        assert truncate_abstracts(series, None) is series
        assert truncate_abstracts(series, 0) is series

    def test_the_cap_removes_the_bulk_of_a_heavy_tailed_corpus(self):
        """The point of the whole exercise, in one assertion."""
        corpus = pd.Series(['x' * 1500] * 90 + ['x' * 25_000] * 10)
        before = corpus.str.len().sum()
        after = truncate_abstracts(corpus, 2500).str.len().sum()
        assert after < before * 0.45
