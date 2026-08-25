"""Tests for the review batch parser's resilience to a missing `reasoning` field.

Some providers/models return the structured evaluation without the optional
`reasoning` key. The parser must default it to "" rather than raise KeyError
(which previously bubbled up as a RuntimeError and killed the whole review).
"""

import asyncio

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

    def test_missing_evaluation_still_fails(self):
        # evaluation is the score — it stays strict; a missing one must not be
        # silently defaulted, so the reviewer still raises after its retries.
        with pytest.raises(RuntimeError):
            _run(_reviewer({"reasoning": "no score"}), ["t"])
