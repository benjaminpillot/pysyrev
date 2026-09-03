"""Tests for how a failing review call is recovered.

A batch can fail for three reasons that look alike in the logs and need opposite
responses. Splitting the batch — the recovery written for a model that drops
items — is actively harmful for the other two:

* a rate limit (429) is the endpoint asking for fewer requests, so a split,
  which sends more, turns one refusal into a cascade of them;
* a truncation is the answer being cut off at ``max_tokens``, and since that
  budget is multiplied by the batch size, each half of a split gets exactly the
  same budget per article and is cut off at the same place.

These tests pin the classification and the two recoveries that replace the split
(wait-and-resend, budget doubling), plus the client-side pacing that keeps a
per-minute quota from being spent in the first two seconds of a chunk.
"""

import asyncio
import re
import time

import pytest

from pysyrev.core import llm
from pysyrev.core.llm import (ALBERT_API_KEY_ENV, ALBERT_REQUESTS_PER_MINUTE,
                              TruncatedResponse, _extract_json, _is_rate_limited,
                              _RateLimiter, _review_batch, Reviewer,
                              build_reviewer)


# ── Fakes ──────────────────────────────────────────────────────────────────

class _RateLimitError(Exception):
    """Stand-in for openai.RateLimitError as Albert raises it."""
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - {'detail': '10 requests per minute "
                         "exceeded (remaining: 0).'}")


class _RecordingProvider:
    """Records every call, then replies from a scripted list.

    A reply is an exception to raise, or ``None`` to answer correctly for the
    batch size that was asked for. The script's last entry repeats forever.
    """

    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []          # (n_articles, max_tokens) per call

    async def call(self, messages, model_id, model_args, response_schema):
        match = re.search(r"exactly (\d+) items", messages[-1]["content"])
        n_articles = int(match.group(1)) if match else 1
        self.calls.append((n_articles, model_args.get("max_tokens")))

        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        if n_articles == 1:
            return {"evaluation": 4, "reasoning": "ok"}, None
        return ({"evaluations": [{"evaluation": 3, "reasoning": "ok"}
                                 for _ in range(n_articles)]}, None)

    @property
    def batch_sizes(self):
        return [n for n, _ in self.calls]

    @property
    def budgets(self):
        return [b for _, b in self.calls]


def _reviewer(provider, *, name="Reviewer#1", max_tokens=200, max_retries=2,
              requests_per_minute=None, provider_name="fake", host=None,
              max_concurrent_requests=1):
    return Reviewer(
        name=name, model_id="m", provider=provider_name,
        provider_client=provider,
        backstory="bs", reasoning=None, max_tokens=max_tokens, temperature=None,
        reasoning_effort=None, additional_context=None,
        inclusion_criteria="inc", exclusion_criteria="exc",
        input_description="article title/abstract/keywords",
        max_retries=max_retries,
        max_concurrent_requests=max_concurrent_requests, items_per_call=5,
        requests_per_minute=requests_per_minute, host=host,
    )


def _run(reviewer, texts, limiter=None):
    async def _call():
        return await _review_batch(texts, reviewer, asyncio.Semaphore(1), limiter)
    return asyncio.run(_call())


@pytest.fixture(autouse=True)
def fresh_limiters():
    """The limiter registry is process-wide; no test may inherit another's window."""
    llm._LIMITERS.clear()
    yield
    llm._LIMITERS.clear()


@pytest.fixture
def instant_backoff(monkeypatch):
    """Keep the 429 waits out of the test clock."""
    monkeypatch.setattr(llm, "RATE_LIMIT_BASE_DELAY", 0.001)
    monkeypatch.setattr(llm, "RATE_LIMIT_MAX_DELAY", 0.01)


# ── Truncation ─────────────────────────────────────────────────────────────

class TestTruncation:
    """A cut-off answer is a budget problem, so it buys a bigger budget."""

    def test_the_reported_json_error_is_recognised_as_a_truncation(self):
        # The exact shape that produced "Expecting ',' delimiter: line 18
        # column 6": the response stops after a complete item, so the greedy
        # fallback regex matches up to that item's closing brace and the decoder
        # reports a delimiter error deep inside a response that is merely cut off.
        cut_off = ('{\n  "evaluations": [\n'
                   '    {\n      "evaluation": 3,\n      "reasoning": "a"\n    },\n'
                   '    {\n      "evaluation": 4,\n      "reasoning": "b"\n    }')
        with pytest.raises(TruncatedResponse):
            _extract_json(cut_off)

    def test_malformed_but_complete_json_is_not_called_a_truncation(self):
        with pytest.raises(ValueError) as excinfo:
            _extract_json('{"evaluations": [{"evaluation": 3,,}]}')
        assert not isinstance(excinfo.value, TruncatedResponse)

    def test_truncation_retries_with_a_doubled_budget(self):
        provider = _RecordingProvider(TruncatedResponse(800), None)
        out = _run(_reviewer(provider), ["a", "b", "c", "d"])
        assert len(out) == 4
        # Same batch re-sent, with twice the budget — never halved.
        assert provider.batch_sizes == [4, 4]
        assert provider.budgets == [800, 1600]

    def test_truncation_does_not_consume_the_retry_budget(self):
        """A call that never produced an answer is not a failed attempt."""
        provider = _RecordingProvider(TruncatedResponse(200), None)
        out = _run(_reviewer(provider, max_retries=1), ["only one"])
        assert out == [{"evaluation": 4, "reasoning": "ok"}]
        assert provider.budgets == [200, 400]

    def test_persistent_truncation_raises_instead_of_splitting(self):
        provider = _RecordingProvider(TruncatedResponse(1000))
        with pytest.raises(RuntimeError, match="max_tokens"):
            _run(_reviewer(provider), ["a", "b", "c", "d", "e"])
        # One call plus one per doubling, and not one of them a split.
        assert provider.batch_sizes == [5] * (1 + llm.TRUNCATION_MAX_BUMPS)

    def test_truncation_with_no_configured_budget_still_splits(self):
        """Nothing here can raise a cap the endpoint owns, but a shorter answer
        may fit under it — so that one keeps the split."""
        class _TruncateBatches(_RecordingProvider):
            async def call(self, messages, model_id, model_args, response_schema):
                match = re.search(r"exactly (\d+) items", messages[-1]["content"])
                if match:
                    self.calls.append((int(match.group(1)), None))
                    raise TruncatedResponse()
                return await super().call(messages, model_id, model_args,
                                          response_schema)

        provider = _TruncateBatches(None)
        out = _run(_reviewer(provider, max_tokens=None), ["a", "b", "c", "d"])
        assert len(out) == 4
        assert 2 in provider.batch_sizes        # it did split


# ── Rate limiting ──────────────────────────────────────────────────────────

class TestRateLimit:
    """A 429 is answered by waiting, never by sending more requests."""

    def test_the_gateway_message_is_classified_as_a_rate_limit(self):
        assert _is_rate_limited(_RateLimitError())
        assert _is_rate_limited(RuntimeError(
            "Error code: 429 - {'detail': '10 requests per minute exceeded'}"))
        assert not _is_rate_limited(ValueError("Expected 5 evaluations, got 4"))

    def test_rate_limited_batch_is_resent_whole(self, instant_backoff):
        provider = _RecordingProvider(_RateLimitError(), None)
        out = _run(_reviewer(provider), ["a", "b", "c", "d", "e"])
        assert len(out) == 5
        assert provider.batch_sizes == [5, 5]        # waited, did not split

    def test_persistent_rate_limit_never_splits(self, instant_backoff):
        provider = _RecordingProvider(_RateLimitError())
        with pytest.raises(RuntimeError, match="rate-limited"):
            _run(_reviewer(provider), ["a", "b", "c", "d", "e"])
        assert provider.batch_sizes == [5] * (1 + llm.RATE_LIMIT_MAX_WAITS)

    def test_a_429_holds_back_the_reviewer_s_other_calls(self, instant_backoff):
        """The quota belongs to the key, so one call's 429 pauses them all."""
        limiter = _RateLimiter(1000)                  # not the constraint here
        provider = _RecordingProvider(_RateLimitError(), None)
        _run(_reviewer(provider), ["a", "b"], limiter)
        assert limiter._blocked_until > 0.0

    def test_retry_after_header_is_honoured(self):
        class _WithHeader(_RateLimitError):
            class response:
                headers = {"retry-after": "7"}

        assert llm._rate_limit_delay(_WithHeader(), 0) == 7.0

    def test_backoff_is_capped_at_one_window(self):
        assert llm._rate_limit_delay(_RateLimitError(), 10) <= \
            llm.RATE_LIMIT_MAX_DELAY * 1.1


# ── Timeouts ───────────────────────────────────────────────────────────────

class _Timeout(Exception):
    """Stand-in for openai.APITimeoutError."""

    def __init__(self):
        super().__init__("Request timed out.")


class TestTimeout:
    """An endpoint that did not answer is not asked twice as hard."""

    def test_the_sdk_message_is_classified_as_a_timeout(self):
        assert llm._is_timeout(_Timeout())
        assert llm._is_timeout(asyncio.TimeoutError())
        assert not llm._is_timeout(_RateLimitError())
        assert not llm._is_timeout(ValueError("Expected 5 evaluations, got 4"))

    def test_a_timed_out_batch_is_resent_whole(self):
        provider = _RecordingProvider(_Timeout(), None)
        out = _run(_reviewer(provider), ["a", "b", "c", "d", "e"])
        assert len(out) == 5
        assert provider.batch_sizes == [5, 5]        # re-sent, never halved

    def test_a_persistent_timeout_never_splits(self):
        """Halving would aim two unanswered requests where one already failed,
        and each half would sit out the whole timeout again."""
        provider = _RecordingProvider(_Timeout())
        with pytest.raises(RuntimeError, match="did not answer"):
            _run(_reviewer(provider), ["a", "b", "c", "d", "e"])
        assert provider.batch_sizes == [5] * (1 + llm.TIMEOUT_MAX_RETRIES)

    def test_a_timeout_does_not_consume_the_retry_budget(self):
        provider = _RecordingProvider(_Timeout(), None)
        out = _run(_reviewer(provider, max_retries=1), ["only one"])
        assert out == [{"evaluation": 4, "reasoning": "ok"}]


class TestRateLimiter:

    def test_calls_beyond_the_limit_wait_for_the_window(self):
        limiter = _RateLimiter(2, window=0.15)

        async def _spend(n):
            start = time.monotonic()
            for _ in range(n):
                await limiter.acquire()
            return time.monotonic() - start

        assert asyncio.run(_spend(2)) < 0.1          # first window is free
        assert asyncio.run(_spend(1)) >= 0.05        # third call had to wait

    def test_concurrent_callers_share_one_quota(self):
        limiter = _RateLimiter(2, window=0.15)

        async def _race():
            start = time.monotonic()
            await asyncio.gather(*(limiter.acquire() for _ in range(4)))
            return time.monotonic() - start

        # Four calls against a quota of two must span more than one window,
        # however many of them are launched at once.
        assert asyncio.run(_race()) >= 0.1


class TestSharedPacing:
    """One quota, one limiter — whoever is spending it, and whenever.

    The quota is metered per API key. A limiter scoped to one reviewer, or to
    one checkpoint chunk, therefore starts every turn believing the window is
    empty and opens with a full-quota burst on top of the burst the previous
    turn has just spent — which is how a paced run still collected 429s.
    """

    def _albert(self, name, **kwargs):
        return _reviewer(_RecordingProvider(None), name=name,
                         provider_name="albert", requests_per_minute=10,
                         **kwargs)

    def test_reviewers_on_one_endpoint_share_a_limiter(self):
        first, second = self._albert("Reviewer#1"), self._albert("Reviewer#2")
        assert llm._rate_limiter_for(first) is llm._rate_limiter_for(second)

    def test_a_separate_endpoint_keeps_its_own_quota(self):
        assert llm._rate_limiter_for(self._albert("R", host="https://a/v1")) \
            is not llm._rate_limiter_for(self._albert("R", host="https://b/v1"))

    def test_the_strictest_pace_configured_wins(self):
        llm._rate_limiter_for(self._albert("Reviewer#1"))
        limiter = llm._rate_limiter_for(
            _reviewer(_RecordingProvider(None), provider_name="albert",
                      requests_per_minute=3))
        assert limiter._limit == 3

    def test_an_unpaced_reviewer_gets_no_limiter(self):
        assert llm._rate_limiter_for(_reviewer(_RecordingProvider(None))) is None

    def test_the_quota_is_spent_when_the_call_actually_goes_out(self):
        """A slot taken while queued behind the semaphore stamps a send that has
        not happened; the real request then lands in the next window, on top of
        the calls that window had already budgeted for."""
        limiter = _RateLimiter(10, window=30)     # never the constraint here
        semaphore = asyncio.Semaphore(1)          # concurrency is

        class _Slow(_RecordingProvider):
            async def call(self, *args):
                await asyncio.sleep(0.15)
                return await super().call(*args)

        reviewer = _reviewer(_Slow(None))

        async def _three_at_once():
            await asyncio.gather(*(_review_batch(["a"], reviewer, semaphore,
                                                 limiter)
                                   for _ in range(3)))

        asyncio.run(_three_at_once())
        stamps = list(limiter._started)
        assert len(stamps) == 3
        assert stamps[-1] - stamps[0] >= 0.25     # spread over the real sends

    def test_a_limiter_survives_the_per_chunk_event_loop(self):
        """`process_per_batch` opens one `asyncio.run` per checkpoint chunk.

        Two things used to break at that boundary: the window forgot the calls
        of the previous chunk (which `api_pause` is too short to cover), and a
        lock built in the closed loop raised as soon as it was awaited again.
        """
        limiter = _RateLimiter(2, window=0.3)
        asyncio.run(limiter.acquire())
        asyncio.run(limiter.acquire())

        start = time.monotonic()
        asyncio.run(limiter.acquire())
        assert time.monotonic() - start >= 0.05

    def test_the_second_reviewer_of_a_round_waits_out_the_first(self, monkeypatch):
        """A round runs its reviewers one after another, on the same key."""
        monkeypatch.setattr(llm, "RATE_LIMIT_WINDOW", 0.3)
        texts = [f"a{i}" for i in range(10)]         # 2 calls of 5 per reviewer
        reviewers = [_reviewer(_RecordingProvider(None), name=name,
                               provider_name="albert", requests_per_minute=2,
                               max_concurrent_requests=10)
                     for name in ("Reviewer#1", "Reviewer#2")]

        async def _round():
            start = time.monotonic()
            for reviewer in reviewers:
                await llm._review_all(texts, reviewer)
            return time.monotonic() - start

        # Four calls against a quota of two: the second reviewer cannot start
        # until the first one's window has slid, however fast its own calls are.
        assert asyncio.run(_round()) >= 0.25


class TestLabellerPacing:
    """The topic and community labellers spend the reviewers' quota."""

    def _config(self, provider="albert", **kwargs):
        from pysyrev.core.config import TopicLabelerConfig
        return TopicLabelerConfig(provider=provider, model_id="m", **kwargs)

    def test_the_labeller_draws_on_the_reviewers_limiter(self):
        reviewer = _reviewer(_RecordingProvider(None), provider_name="albert",
                             requests_per_minute=10)
        assert llm._labeler_limiter(self._config()) \
            is llm._rate_limiter_for(reviewer)

    def test_an_unpaced_provider_still_gets_none(self):
        assert llm._labeler_limiter(self._config(provider="anthropic")) is None

    def test_a_429_waits_instead_of_burning_an_attempt(self, instant_backoff,
                                                       monkeypatch, capsys):
        """It used to retry immediately, spend both attempts and lose the label."""
        class _Labeller:
            calls = 0

            async def call(self, messages, model_id, model_args, schema):
                _Labeller.calls += 1
                if _Labeller.calls == 1:
                    raise _RateLimitError()
                return {"label": "Solar PV adoption"}, None

        monkeypatch.setattr(llm, "_make_provider", lambda *a, **k: _Labeller())
        out = llm.label_clusters({0: ["solar", "pv"]},
                                 self._config(max_retries=2))
        assert out == {0: "Solar PV adoption"}
        assert _Labeller.calls == 2
        assert "rate-limited" in capsys.readouterr().out


# ── Pacing defaults ────────────────────────────────────────────────────────

class TestPacingDefaults:

    def _build(self, monkeypatch, provider, **kwargs):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        return build_reviewer(
            name="R", provider=provider, model_id="m", host=None,
            reasoning=None, max_tokens=200, temperature=None,
            reasoning_effort=None, backstory="bs", additional_context=None,
            inclusion_criteria="inc", exclusion_criteria="exc",
            input_description="article title/abstract/keywords", **kwargs)

    def test_albert_is_paced_to_its_published_quota_by_default(self, monkeypatch):
        pytest.importorskip("openai")
        reviewer = self._build(monkeypatch, "albert")
        assert reviewer.requests_per_minute == ALBERT_REQUESTS_PER_MINUTE

    def test_the_config_overrides_the_provider_default(self, monkeypatch):
        pytest.importorskip("openai")
        reviewer = self._build(monkeypatch, "albert", requests_per_minute=4)
        assert reviewer.requests_per_minute == 4

    def test_other_providers_are_not_paced_on_a_guess(self, monkeypatch):
        reviewer = self._build(monkeypatch, "litellm")
        assert reviewer.requests_per_minute is None
