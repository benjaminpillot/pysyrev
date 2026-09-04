"""
What to do when a call comes back wrong, and how not to get there.

Three failure kinds are told apart here because each wants a different answer:
a **truncated** response was cut at the output cap and wants a bigger budget; a
**429** says the key is out of quota and wants a wait, not a retry; a
**timeout** is the endpoint not answering and wants the same request re-sent.

:class:`_RateLimiter` is the "not getting there" half — one shared window per
provider account, so the whole run paces itself against the quota the gateway
actually counts.
"""

import asyncio
import random
import time
from collections import deque
from typing import Dict, Optional


#: How many times a truncated batch is retried with a doubled output budget
#: before giving up on it. Two bumps take max_tokens to 4x what the config asks.
TRUNCATION_MAX_BUMPS = 2

#: How many times a rate-limited call waits and retries the *same* batch.
#: A 429 says "come back later", so the batch must not be split (that only
#: sends more requests at an endpoint already refusing them).
RATE_LIMIT_MAX_WAITS = 5
RATE_LIMIT_BASE_DELAY = 15.0    # seconds, doubled per consecutive 429
RATE_LIMIT_MAX_DELAY = 60.0     # a per-minute window never needs more

#: How many times a timed-out call re-sends the *same* batch. A timeout is the
#: endpoint failing to answer, not the model struggling with the ask, so the
#: split is wrong here for the reason it is wrong for a 429: it aims more
#: requests at something that is already not answering, and each half would sit
#: out the same timeout again before failing in its turn.
TIMEOUT_MAX_RETRIES = 2

#: Length of the window the client paces itself over, for a quota expressed per
#: minute. Deliberately longer than the minute it stands for: the gateway counts
#: a request when it *arrives*, we count it when we start sending, and the
#: difference (network latency, and the gateway's own window boundary) is enough
#: to make the 11th call land inside the server's window while our own has just
#: expired. The margin costs ~9 % throughput and removes a whole class of 429s.
RATE_LIMIT_WINDOW = 66.0


class TruncatedResponse(Exception):
    """The model stopped on its output cap: the JSON is cut off, not malformed.

    Worth its own type because the two generic recoveries are both wrong here.
    Retrying identically re-truncates at the same token, and splitting the batch
    does not help either: ``max_tokens`` is multiplied by the batch size, so the
    budget per article is the same in a batch of 5 and in a batch of 1. The only
    fix is a bigger budget, which is what the caller does with ``max_tokens``.
    """

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        super().__init__(
            f"response truncated at max_tokens={max_tokens} — the model ran out "
            f"of output budget mid-JSON" if max_tokens else
            "response truncated by the endpoint's output cap")


#: finish_reason / stop_reason values that mean "cut off at the output cap".
_TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens"}


def _raise_if_truncated(finish_reason, model_args: dict) -> None:
    """Turn a cut-off completion into :class:`TruncatedResponse`.

    Called before parsing: a truncated response is not a JSON problem, and
    letting it reach the parser is what turned an output-budget error into an
    unreadable ``Expecting ',' delimiter`` further down.
    """
    if finish_reason and str(finish_reason).lower() in _TRUNCATION_REASONS:
        raise TruncatedResponse(model_args.get("max_tokens"))


def _is_rate_limited(exc: Exception) -> bool:
    """Is `exc` the endpoint refusing a request it would accept later?"""
    status = (getattr(exc, "status_code", None)
              or getattr(getattr(exc, "response", None), "status_code", None))
    if status == 429:
        return True
    text = str(exc).lower()
    return any(k in text for k in ("rate limit", "ratelimit", "too many requests",
                                   "per minute exceeded", "per-minute exceeded",
                                   "error code: 429"))


def _is_timeout(exc: Exception) -> bool:
    """Is `exc` the endpoint failing to answer in time?

    Recognised by type where an SDK exposes one (`APITimeoutError`,
    `httpx.TimeoutException`, `asyncio.TimeoutError`) and by message otherwise,
    because litellm re-wraps the transport's exception in its own.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if "timeout" in type(exc).__name__.lower():
        return True
    return "timed out" in str(exc).lower()


def _rate_limit_delay(exc: Exception, waits: int) -> float:
    """Seconds to wait before retrying a rate-limited call.

    ``Retry-After`` is authoritative when the gateway sends it; otherwise back
    off exponentially, capped at one window — a per-minute quota can never need
    longer than the minute it is measured over.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        for key in ("retry-after", "x-ratelimit-reset-requests"):
            try:
                value = headers.get(key)
            except Exception:
                value = None
            if value:
                try:
                    return min(float(value), RATE_LIMIT_MAX_DELAY)
                except (TypeError, ValueError):
                    pass
    delay = min(RATE_LIMIT_BASE_DELAY * (2 ** waits), RATE_LIMIT_MAX_DELAY)
    # Jitter: concurrent callers hit the quota together and must not all come
    # back together, or the first retry wave rebuilds the burst that caused it.
    return delay + random.uniform(0, 0.1 * delay)


class _RateLimiter:
    """Sliding-window limiter: at most `requests_per_minute` calls per window.

    Albert allows 10 requests per minute per key, and ``max_concurrent_requests``
    alone cannot honour that — 10 concurrent calls that each take two seconds
    spend the whole minute's quota in two seconds, and every later call in the
    chunk comes back 429. Concurrency bounds how many calls are *in flight*;
    this bounds how many are *started* per minute, which is what a quota counts.

    Shared by all the calls of one reviewer, so a 429 seen by any of them backs
    every one of them off (:meth:`penalise`) instead of letting the others keep
    hammering a quota that is already exhausted.

    One limiter stands for one quota, so it must also outlive the thing that
    happens to be running: reviewers take their turn one after another inside a
    round, and each checkpoint chunk gets its own ``asyncio.run``. A limiter
    scoped to either would start every turn with an empty window and burst a
    whole minute's quota at an endpoint that has just been served the previous
    turn's — which is the cascade this class exists to prevent. Hence the
    registry in :func:`_rate_limiter_for`, and hence the lock below being
    rebuilt per event loop: the window rides on :func:`time.monotonic`, which
    outlives a loop, while an :class:`asyncio.Lock` does not.
    """

    def __init__(self, requests_per_minute: int,
                 window: Optional[float] = None):
        self._limit = max(1, int(requests_per_minute))
        self._window = RATE_LIMIT_WINDOW if window is None else window
        self._started: deque = deque()
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop = None
        self._blocked_until = 0.0

    def tighten(self, requests_per_minute: int) -> None:
        """Adopt `requests_per_minute` if it is stricter than the current limit.

        Two reviewers on one key may ask for different pacing; the quota only
        knows the sum of their calls, so the smaller number is the honest one.
        """
        self._limit = min(self._limit, max(1, int(requests_per_minute)))

    def _get_lock(self) -> asyncio.Lock:
        """The lock for the running loop, rebuilt when the loop changes.

        A lock created in a closed loop raises as soon as it is awaited in
        another one. Nothing is lost by replacing it: the waiters it guarded
        died with their loop, and the window state is kept outside it.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def acquire(self) -> None:
        while True:
            async with self._get_lock():
                now = time.monotonic()
                while self._started and now - self._started[0] >= self._window:
                    self._started.popleft()

                if self._blocked_until > now:
                    wait = self._blocked_until - now
                elif len(self._started) >= self._limit:
                    wait = self._window - (now - self._started[0])
                else:
                    self._started.append(now)
                    return
            # Slept outside the lock: holding it would serialise every waiter
            # behind the longest wait instead of letting them re-check together.
            await asyncio.sleep(min(wait, self._window) + 0.01)

    async def penalise(self, seconds: float) -> None:
        """Hold back every call drawing on this quota for `seconds` after a 429."""
        async with self._get_lock():
            self._blocked_until = max(self._blocked_until,
                                      time.monotonic() + seconds)


#: Live limiters, one per quota (see :func:`_rate_limiter_for`). Module-level
#: because the quota outlives every scope the review has: the reviewer, the
#: round, the checkpoint chunk and its event loop.
_LIMITERS: Dict[tuple, _RateLimiter] = {}


def _limiter_for_quota(provider: str, host: Optional[str],
                       requests_per_minute: Optional[int]) -> Optional['_RateLimiter']:
    """The shared limiter for one provider account, or None when unpaced.

    What the endpoint counts against is the API key, and a run reads one key per
    provider from the environment — so provider + host identifies the quota as
    precisely as anything available here. Everything that spends it comes
    through this function: the reviewers of every round, and the topic and
    community labellers, which are few calls but land on the same key.

    Shared, not per caller: Albert's ten calls a minute are ten for the whole
    run, not ten for each of the three reviewers a workflow puts on that key.
    Callers take their turn one after another, so a per-caller limiter let each
    one open with a full-quota burst on top of the burst the previous one had
    just spent. The registry also survives the per-chunk ``asyncio.run`` of
    :func:`process_per_batch`, where `api_pause` is shorter than the window it
    would have to cover.
    """
    if not requests_per_minute:
        return None
    key = (provider, host)
    limiter = _LIMITERS.get(key)
    if limiter is None:
        limiter = _LIMITERS[key] = _RateLimiter(requests_per_minute)
    else:
        limiter.tighten(requests_per_minute)
    return limiter
