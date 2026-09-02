"""Tests for the a priori token / cost estimator.

Everything here runs offline (a stub counter, no API key, no network): what is
being tested is the accounting — batching, round fan-out, the price table, the
prompt reconstruction — not Anthropic's tokenizer.
"""

import pandas as pd
import pytest

from pysyrev.core.token_cost import (
    BATCH_DISCOUNT,
    HeuristicTokenCounter,
    PRICING,
    estimate_review,
    fit_article_model,
    resolve_price,
    sweep_items_per_call,
)
from pysyrev.core.config import ReviewConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

def _review_config(**overrides):
    data = dict(
        export={'export_dir': '/tmp/pysyrev-test-estimate'},
        # _input_description indexes three inputs, as every real config does.
        text_inputs=['title', 'abstract', 'keywords'],
        inclusion_criteria='must be about X',
        exclusion_criteria='must not be about Y',
        doc_dataset=None,
        items_per_call=10,
        reviewers=[
            dict(name='Reviewer#1', model_id='claude-haiku-4-5', host=None,
                 provider='anthropic', max_tokens=200, temperature=0.5,
                 reasoning_effort=None, reasoning='brief',
                 backstory='an expert', additional_context=None),
            dict(name='Reviewer#2', model_id='claude-haiku-4-5', host=None,
                 provider='anthropic', max_tokens=200, temperature=0.5,
                 reasoning_effort=None, reasoning='brief',
                 backstory='another expert', additional_context=None),
        ],
        workflow=[
            {'round': 'A', 'reviewers': ['Reviewer#1']},
            {'round': 'B', 'reviewers': ['Reviewer#2']},
        ],
    )
    data.update(overrides)
    return ReviewConfig(**data)


def _dataset(n=100, abstract_chars=1000, keywords=True):
    data = {
        'title':    [f'Title of paper {i}' for i in range(n)],
        'abstract': ['a' * abstract_chars for _ in range(n)],
    }
    if keywords:
        data['keywords'] = ['energy; model; transition' for _ in range(n)]
    return pd.DataFrame(data)


def _estimate(**kwargs):
    kwargs.setdefault('offline', True)
    kwargs.setdefault('sample_size', 4)
    return estimate_review(_review_config(), _dataset(), **kwargs)


# ── Pricing ────────────────────────────────────────────────────────────────

class TestPricing:

    def test_plain_model_id(self):
        assert resolve_price('claude-haiku-4-5') is PRICING['claude-haiku-4-5']

    def test_dated_snapshot_resolves_to_base_model(self):
        assert resolve_price('claude-haiku-4-5-20251001') is PRICING['claude-haiku-4-5']

    def test_litellm_prefix_is_stripped(self):
        assert resolve_price('anthropic/claude-haiku-4-5') is PRICING['claude-haiku-4-5']

    def test_unknown_model_names_the_known_ones(self):
        with pytest.raises(KeyError, match='claude-haiku-4-5'):
            resolve_price('gpt-4o-mini')


# ── Per-article token model ────────────────────────────────────────────────

class _Linear(HeuristicTokenCounter):
    """A counter that is exactly ``10 + len(user)/4``."""

    def count(self, system, user, tools=None, tool_choice=None):
        return 10 + len(user) // 4


def _measure(texts, sample_size=8):
    return fit_article_model(texts, _Linear(),
                             lambda ts: ('sys', ''.join(ts), None, None),
                             sample_size=sample_size)


class TestArticleModel:

    def test_total_matches_a_known_linear_counter(self):
        texts = ['x' * n for n in (100, 500, 2000, 8000)]
        model = _measure(texts, sample_size=4)
        assert model.total == pytest.approx(sum(len(t) for t in texts) / 4, rel=0.02)

    def test_heavy_tail_is_not_averaged_away(self):
        """One 100k-character outlier among short texts must reach the total."""
        texts = ['x' * 200] * 99 + ['x' * 100_000]
        model = _measure(texts, sample_size=16)
        assert model.total == pytest.approx((99 * 200 + 100_000) / 4, rel=0.05)

    def test_a_degenerate_length_distribution_is_still_exact(self):
        """Truncating every text to one length used to break the linear fit."""
        model = _measure(['x' * 2500] * 200, sample_size=16)
        assert model.total == pytest.approx(200 * 2500 / 4, rel=0.01)

    def test_totals_are_monotonic_in_text_length(self):
        short = _measure(['x' * n for n in range(100, 2600, 25)], sample_size=16)
        long_ = _measure(['x' * n for n in range(100, 5100, 50)], sample_size=16)
        assert long_.total > short.total


# ── Estimate accounting ────────────────────────────────────────────────────

class TestEstimate:

    def test_first_round_covers_the_whole_corpus(self):
        est = _estimate(escalation_rate=0.1)
        round_a = [g for g in est.groups if g.round_name == 'A']
        assert [g.n_items for g in round_a] == [100]

    def test_later_rounds_are_scaled_by_the_escalation_rate(self):
        est = _estimate(escalation_rate=0.2)
        round_b = next(g for g in est.groups if g.round_name == 'B')
        assert round_b.n_items == 20

    def test_a_zero_escalation_rate_drops_later_rounds(self):
        est = _estimate(escalation_rate=0.0)
        assert {g.round_name for g in est.groups} == {'A'}

    def test_calls_follow_items_per_call(self):
        est = _estimate()
        round_a = next(g for g in est.groups if g.round_name == 'A')
        assert round_a.items_per_call == 10
        assert round_a.n_calls == 10          # 100 articles / 10 per call

    def test_batch_api_halves_the_cost(self):
        est = _estimate()
        assert est.cost(batch=True) == pytest.approx(BATCH_DISCOUNT * est.cost())

    def test_output_tokens_are_capped_by_max_tokens(self):
        est = estimate_review(_review_config(), _dataset(),
                              offline=True, sample_size=4,
                              output_tokens={'Reviewer#1': 10_000})
        round_a = next(g for g in est.groups if g.round_name == 'A')
        assert round_a.capped
        assert round_a.output_tokens == 200 * 10 * 10   # max_tokens × per call × calls

    def test_longer_abstracts_cost_more(self):
        cheap = estimate_review(_review_config(), _dataset(abstract_chars=500),
                                offline=True, sample_size=4)
        pricey = estimate_review(_review_config(), _dataset(abstract_chars=5000),
                                 offline=True, sample_size=4)
        assert pricey.input_tokens > 5 * cheap.input_tokens

    def test_missing_text_inputs_are_reported(self):
        """A text_input with no matching column is silently dropped at run time."""
        est = estimate_review(_review_config(), _dataset(keywords=False),
                              offline=True, sample_size=4)
        assert any('keywords' in note for note in est.notes)

    def test_heuristic_mode_is_flagged(self):
        est = _estimate()
        assert est.exact is False
        assert 'heuristic' in est.render().lower()

    def test_no_dataset_and_no_doc_dataset_is_an_error(self):
        with pytest.raises(ValueError, match='No dataset'):
            estimate_review(_review_config(), None, offline=True)


# ── Sweep ──────────────────────────────────────────────────────────────────

class TestSweep:

    def test_sweep_is_monotonically_cheaper_with_bigger_batches(self):
        est = _estimate()
        table = sweep_items_per_call(est, values=(1, 10, 50))
        costs = [float(line.split()[3]) for line in table.splitlines()[2:]]
        assert costs == sorted(costs, reverse=True)
