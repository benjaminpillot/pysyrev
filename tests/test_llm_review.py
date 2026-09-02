"""Tests for the review batch parser's resilience to a missing `reasoning` field.

Some providers/models return the structured evaluation without the optional
`reasoning` key. The parser must default it to "" rather than raise KeyError
(which previously bubbled up as a RuntimeError and killed the whole review).
"""

import asyncio
import re

import pytest

from pysyrev.core.llm import _review_batch, Reviewer


class _FakeProvider:
    def __init__(self, raw):
        self._raw = raw

    async def call(self, messages, model_id, model_args, response_schema):
        return self._raw, None


def _reviewer(raw):
    return Reviewer(
        name="Reviewer#3", model_id="m", provider="fake",
        provider_client=_FakeProvider(raw),
        backstory="bs", reasoning=None, max_tokens=None, temperature=None,
        reasoning_effort=None, additional_context=None,
        inclusion_criteria="inc", exclusion_criteria="exc",
        input_description="article title/abstract/keywords",
        max_retries=2, max_concurrent_requests=1, items_per_call=1,
    )


def _run(reviewer, texts):
    async def _call():
        return await _review_batch(texts, reviewer, asyncio.Semaphore(1))
    return asyncio.run(_call())


class TestReasoningResilience:

    def test_single_missing_reasoning_defaults_to_empty(self):
        out = _run(_reviewer({"evaluation": 4}), ["some article text"])
        assert out == [{"evaluation": 4, "reasoning": ""}]

    def test_batch_missing_reasoning_defaults_to_empty(self):
        raw = {"evaluations": [{"evaluation": 5}, {"evaluation": 2}]}
        out = _run(_reviewer(raw), ["t1", "t2"])
        assert out == [{"evaluation": 5, "reasoning": ""},
                       {"evaluation": 2, "reasoning": ""}]

    def test_reasoning_kept_when_present(self):
        out = _run(_reviewer({"evaluation": 3, "reasoning": "because"}), ["t"])
        assert out == [{"evaluation": 3, "reasoning": "because"}]

    def test_single_item_wrapped_in_batch_envelope_is_unwrapped(self):
        # The real crash: a single-item call whose model returns the batch
        # envelope {"evaluations": [item]} instead of the bare item — used to
        # raise KeyError('evaluation'). It must unwrap to the sole item.
        raw = {"evaluations": [{"evaluation": 5, "reasoning": "wrapped"}]}
        out = _run(_reviewer(raw), ["one article"])
        assert out == [{"evaluation": 5, "reasoning": "wrapped"}]

    def test_single_item_as_bare_list_is_unwrapped(self):
        raw = [{"evaluation": 2, "reasoning": "listed"}]
        out = _run(_reviewer(raw), ["one article"])
        assert out == [{"evaluation": 2, "reasoning": "listed"}]

    def test_missing_evaluation_still_fails(self):
        # evaluation is the score — it stays strict; a missing one must not be
        # silently defaulted, so the reviewer still raises after its retries.
        with pytest.raises(RuntimeError):
            _run(_reviewer({"reasoning": "no score"}), ["t"])


class _DropOneProvider:
    """Returns one fewer evaluation than requested for any multi-item batch,
    forcing the count-mismatch path; single-item calls (no 'exactly N' in the
    prompt) succeed. Models the real failure: large batches drop items."""

    def __init__(self):
        self.calls = 0

    async def call(self, messages, model_id, model_args, response_schema):
        self.calls += 1
        user = messages[-1]["content"]
        m = re.search(r"exactly (\d+) items", user)
        if not m:                       # single-item schema path — reliable
            return {"evaluation": 4, "reasoning": "single"}, None
        n = int(m.group(1))
        evals = [{"evaluation": 3, "reasoning": "r"} for _ in range(n - 1)]
        return {"evaluations": evals}, None


def _reviewer_with(provider, *, items_per_call=50, max_retries=2):
    return Reviewer(
        name="Reviewer#1", model_id="m", provider="fake",
        provider_client=provider,
        backstory="bs", reasoning=None, max_tokens=None, temperature=None,
        reasoning_effort=None, additional_context=None,
        inclusion_criteria="inc", exclusion_criteria="exc",
        input_description="article title/abstract/keywords",
        max_retries=max_retries, max_concurrent_requests=1,
        items_per_call=items_per_call,
    )


class TestBatchCountMismatchSplit:
    """A batch whose model drops items must not crash the run: it splits down to
    the single-item schema and returns one aligned evaluation per article."""

    def test_split_recovers_full_count(self):
        provider = _DropOneProvider()
        reviewer = _reviewer_with(provider)
        texts = [f"article {i}" for i in range(4)]
        out = _run(reviewer, texts)
        assert len(out) == len(texts)                 # one eval per article
        assert all("evaluation" in e for e in out)

    def test_order_preserved_after_split(self):
        # The single-item path tags reasoning "single"; every article resolves
        # through it after the recursive split, so all four are present in order.
        provider = _DropOneProvider()
        out = _run(_reviewer_with(provider), ["a", "b", "c", "d"])
        assert [e["reasoning"] for e in out] == ["single"] * 4

    def test_single_persistent_failure_still_raises(self):
        # A lone article that keeps failing is a genuine error, not a split case.
        class _AlwaysBad:
            async def call(self, messages, model_id, model_args, schema):
                raise RuntimeError("api down")
        with pytest.raises(RuntimeError):
            _run(_reviewer_with(_AlwaysBad(), items_per_call=1), ["only one"])


class _SizeRecordingProvider(_DropOneProvider):
    """_DropOneProvider that records the article count of every call it gets."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    async def call(self, messages, model_id, model_args, response_schema):
        m = re.search(r"exactly (\d+) items", messages[-1]["content"])
        self.batch_sizes.append(int(m.group(1)) if m else 1)
        return await super().call(messages, model_id, model_args, response_schema)


class TestRetriesAreNotBilledTwiceOnBatches:
    """Every attempt at a multi-article batch re-sends all its articles at full
    price, and the dominant failure — the model dropping items — is not fixed by
    an identical retry. So a failing batch is attempted once and then split."""

    def test_a_failing_batch_is_attempted_once_before_splitting(self):
        provider = _SizeRecordingProvider()
        _run(_reviewer_with(provider, max_retries=2), ["a", "b", "c", "d"])
        # 4 fails -> split 2 + 2, each fails once -> split to singles.
        # Retrying each node max_retries times would give [4, 4, 2, 2, 2, 2].
        assert [s for s in provider.batch_sizes if s > 1] == [4, 2, 2]

    def test_batch_call_count_does_not_grow_with_max_retries(self):
        """max_retries must not multiply what a failing batch costs."""
        observed = []
        for retries in (1, 2, 5):
            provider = _SizeRecordingProvider()
            _run(_reviewer_with(provider, max_retries=retries), ["a", "b", "c", "d"])
            observed.append([s for s in provider.batch_sizes if s > 1])
        assert observed == [[4, 2, 2]] * 3

    def test_single_article_calls_keep_the_full_retry_budget(self):
        """Transient failures on a lone article are cheap — still retried."""
        class _CountingBad:
            def __init__(self):
                self.calls = 0

            async def call(self, messages, model_id, model_args, schema):
                self.calls += 1
                raise RuntimeError("transient")

        provider = _CountingBad()
        with pytest.raises(RuntimeError):
            _run(_reviewer_with(provider, items_per_call=1, max_retries=3), ["one"])
        assert provider.calls == 3
