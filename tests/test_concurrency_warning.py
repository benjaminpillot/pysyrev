"""Tests for the batch_size / concurrency consistency check.

items_per_call x max_concurrent_requests is the smallest checkpoint chunk that
fills every concurrency slot. Below it, slots idle for the whole run — a pitfall
the example config documents in prose and that nothing used to enforce.
"""

import pytest

from pysyrev.core.llm import Reviewer
from pysyrev.review import _warn_starved_concurrency


def _reviewer(name='Reviewer#1', items_per_call=25, max_concurrent_requests=10):
    return Reviewer(
        name=name, model_id='claude-haiku-4-5', provider='anthropic',
        provider_client=None, backstory='bs', reasoning='brief',
        max_tokens=200, temperature=0.2, reasoning_effort=None,
        additional_context=None, inclusion_criteria='inc',
        exclusion_criteria='exc', input_description='article',
        max_retries=2, max_concurrent_requests=max_concurrent_requests,
        items_per_call=items_per_call,
    )


class TestStarvedConcurrencyWarning:

    def test_batch_size_at_the_floor_is_silent(self, capsys):
        _warn_starved_concurrency([_reviewer()], batch_size=250)
        assert capsys.readouterr().out == ''

    def test_batch_size_above_the_floor_is_silent(self, capsys):
        _warn_starved_concurrency([_reviewer()], batch_size=500)
        assert capsys.readouterr().out == ''

    def test_batch_size_below_the_floor_warns_with_the_numbers(self, capsys):
        _warn_starved_concurrency([_reviewer()], batch_size=100)
        out = capsys.readouterr().out
        assert 'Reviewer#1' in out
        assert '4 call(s)' in out        # 100 / 25
        assert '6 slot(s) stay' in out   # 10 - 4
        assert '250' in out              # the suggested floor

    def test_a_blank_batch_size_is_a_single_shot_and_never_warns(self, capsys):
        for value in (None, 0):
            _warn_starved_concurrency([_reviewer()], batch_size=value)
        assert capsys.readouterr().out == ''

    def test_each_reviewer_is_checked_on_its_own_overrides(self, capsys):
        reviewers = [
            _reviewer('Reviewer#1', items_per_call=25, max_concurrent_requests=10),
            _reviewer('Reviewer#2', items_per_call=5, max_concurrent_requests=4),
        ]
        _warn_starved_concurrency(reviewers, batch_size=100)
        out = capsys.readouterr().out
        assert 'Reviewer#1' in out       # floor 250 > 100
        assert 'Reviewer#2' not in out   # floor  20 < 100

    def test_a_chunk_smaller_than_one_call_still_reports_one_in_flight(self, capsys):
        """batch_size < items_per_call collapses the chunk into a single call."""
        _warn_starved_concurrency([_reviewer(items_per_call=25)], batch_size=10)
        assert '1 call(s)' in capsys.readouterr().out
