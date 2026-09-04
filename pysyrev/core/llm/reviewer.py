"""
The reviewer: who screens, with what prompt, and how one call is made.

A :class:`Reviewer` bundles a persona (backstory, criteria, reasoning style)
with the provider client and the pacing settings that call is subject to.
Below it, :func:`_prepare_call` builds one request and
:func:`_review_batch` sends it — with the retry, split, budget-bump and
rate-limit recovery that make a round survive a gateway having a bad minute.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from tqdm.asyncio import tqdm

from pysyrev.core.llm.providers import (_BaseProvider, _ReviewBatch,
                                        _ReviewItem,
                                        _default_requests_per_minute,
                                        _make_provider)
from pysyrev.core.llm.ratelimit import (RATE_LIMIT_MAX_WAITS,
                                        TIMEOUT_MAX_RETRIES,
                                        TRUNCATION_MAX_BUMPS,
                                        TruncatedResponse, _RateLimiter,
                                        _is_rate_limited, _is_timeout,
                                        _limiter_for_quota, _rate_limit_delay)


MAX_RETRIES = 2
MAX_CONCURRENT_REQUESTS = 10
ITEMS_PER_CALL = 1

REVIEW_SCORE: str = "review_score"

_REASONING_MAP = {
    None:    "",
    "brief": ("Provide a brief (1-sentence) explanation for your scoring. "
              "State your reasoning before giving the score."),
    "cot":   ("Provide a detailed, step-by-step explanation for your scoring. "
              "State your reasoning before giving the score."),
}


@dataclass
class Reviewer:
    """Runtime reviewer — all parameters needed for one LLM caller."""
    name:                    str
    model_id:                str
    provider:                str           # kept for display / logging
    provider_client:         _BaseProvider  # runtime provider, built by build_reviewer
    backstory:               str
    reasoning:               Optional[str]
    max_tokens:              Optional[int]
    temperature:             Optional[float]
    reasoning_effort:        Optional[str]
    additional_context:      Optional[str]
    inclusion_criteria:      str
    exclusion_criteria:      str
    input_description:       str
    max_retries:             int
    max_concurrent_requests: int
    items_per_call:          int
    #: Calls started per minute, or None for no client-side pacing. Defaults
    #: last, so tests and programmatic callers keep the old signature.
    requests_per_minute:     Optional[int] = None
    #: Endpoint this reviewer talks to. Kept because provider + host is what
    #: identifies the quota the pacing has to share (see `_quota_key`).
    host:                    Optional[str] = None


def build_reviewer(name, provider, model_id,
                   host, reasoning, max_tokens,
                   temperature, reasoning_effort,
                   backstory, additional_context,
                   inclusion_criteria, exclusion_criteria,
                   input_description,
                   **kwargs) -> Reviewer:
    requests_per_minute = (kwargs.get("requests_per_minute")
                           or _default_requests_per_minute(provider))
    provider_client = _make_provider(provider, host, kwargs.get("timeout"))
    # The client retries some requests itself; those must draw on the same
    # quota as everything else, so it is given the shared limiter here.
    provider_client.limiter = _limiter_for_quota(provider, host,
                                                 requests_per_minute)
    return Reviewer(
        name                    = name,
        model_id                = model_id,
        provider                = provider,
        provider_client         = provider_client,
        backstory               = backstory,
        reasoning               = reasoning,
        max_tokens              = max_tokens,
        temperature             = temperature,
        reasoning_effort        = reasoning_effort,
        additional_context      = additional_context,
        inclusion_criteria      = inclusion_criteria,
        exclusion_criteria      = exclusion_criteria,
        input_description       = input_description,
        max_retries             = kwargs.get("max_retries")             or MAX_RETRIES,
        max_concurrent_requests = kwargs.get("max_concurrent_requests") or MAX_CONCURRENT_REQUESTS,
        items_per_call          = kwargs.get("items_per_call")          or ITEMS_PER_CALL,
        requests_per_minute     = requests_per_minute,
        host                    = host,
    )


# ── Prompt builders ────────────────────────────────────────────────────────

def _system_prompt(r: Reviewer) -> str:
    return (
        f"Your name is: <<{r.name}>>. "
        f"Your backstory is: <<{r.backstory}>>. "
        f"Your task is to review {r.input_description}. "
        f"For each article, output an evaluation (integer 1–5, where "
        f"1 = definitely exclude and 5 = definitely include) and a reasoning. "
        f"Inclusion criteria: <<{r.inclusion_criteria}>>. "
        f"Exclusion criteria: <<{r.exclusion_criteria}>>."
    )


def _user_prompt(texts: List[str], r: Reviewer) -> str:
    reasoning = _REASONING_MAP.get(r.reasoning, "") if r.reasoning else ""
    ctx = (f"Use the following additional context: <<{r.additional_context}>>"
           if r.additional_context else "")

    if len(texts) == 1:
        parts = [p for p in [reasoning, ctx, texts[0]] if p]
        return "\n\n".join(parts)

    n_articles = len(texts)
    header = [
        f"Review the following {n_articles} articles in order. "
        f"Return a JSON object with an 'evaluations' array of exactly {n_articles} items. "
        f"Each item must have exactly two keys: "
        f"\"evaluation\" (integer 1-5, where 1 = definitely exclude and 5 = definitely include) "
        f"and \"reasoning\" (string).",
    ]
    if reasoning:
        header.append(reasoning)
    if ctx:
        header.append(ctx)
    articles = "\n\n".join(f"--- Article {i + 1} ---\n{t}" for i, t in enumerate(texts))
    return "\n\n".join(header) + "\n\n" + articles


def _build_model_args(r: Reviewer) -> dict:
    args = {}
    if r.max_tokens:
        args["max_tokens"] = r.max_tokens
    if r.temperature is not None:
        args["temperature"] = r.temperature
    if r.reasoning_effort:
        args["reasoning_effort"] = r.reasoning_effort
    return args


def _normalize_evaluations(raw) -> list:
    """Coerce a provider response into a flat list of evaluation items.

    Providers are inconsistent about the envelope even under structured output:
    a review may come back as a bare list, as ``{"evaluations": [...]}``, or —
    for a single-item call — as the item dict itself (sometimes still wrapped in
    the batch envelope). This normalises all of those to a list of item dicts so
    the count logic and per-item parsing are shape-agnostic. Only the structure
    is unwrapped here; the item contents are validated by :func:`_parse_evaluation`.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("evaluations"), list):
            return raw["evaluations"]
        return [raw]                        # a single item dict
    raise ValueError(f"Unexpected response type: {type(raw)}")


def _parse_evaluation(item: dict) -> dict:
    """Parse one evaluation item into ``{"evaluation": int, "reasoning": str}``.

    ``evaluation`` is the score and stays strict — a missing or non-integer score
    raises (it must never be silently defaulted). ``reasoning`` is optional and
    defaults to an empty string.
    """
    if not isinstance(item, dict):
        raise ValueError(f"Evaluation item is not an object: {type(item)}")
    return {"evaluation": int(item["evaluation"]),
            "reasoning": str(item.get("reasoning", ""))}


def _prepare_call(texts: List[str], reviewer: Reviewer) -> tuple:
    """Everything one review call is made of: ``(messages, model_args, schema)``.

    Sole definition of a review request, so the synchronous and the deferred
    transports send the same thing — and so the cost estimator, which imports
    it, keeps pricing what is actually sent.

    ``max_tokens`` is a per-article budget in the config and is multiplied by
    the batch size here: that is why splitting a truncated batch cannot help,
    since each half is re-budgeted to the same amount per article.
    """
    n_articles = len(texts)
    messages = [
        {"role": "system", "content": _system_prompt(reviewer)},
        {"role": "user",   "content": _user_prompt(texts, reviewer)},
    ]

    model_args = _build_model_args(reviewer)
    if n_articles == 1:
        response_schema = _ReviewItem   # structured output enforces the exact schema
    else:
        response_schema = _ReviewBatch  # structured output enforces key names for all providers
        if "max_tokens" in model_args:
            model_args = {**model_args, "max_tokens": model_args["max_tokens"] * n_articles}
    return messages, model_args, response_schema


def _evaluations_from_raw(raw, n_articles: int) -> List[dict]:
    """Validate and parse a provider response into ``n_articles`` evaluations.

    A single-item call should yield one item, but a model may ignore the single
    schema and answer with the batch envelope or a bare list;
    :func:`_normalize_evaluations` unwraps those. A count that still does not
    match is the dominant failure at large batch sizes — the model silently
    dropping articles — and must raise, because the caller aligns these
    evaluations positionally with the rows it sent.
    """
    evals = _normalize_evaluations(raw)
    if len(evals) != n_articles:
        raise ValueError(f"Expected {n_articles} evaluation(s), got {len(evals)}")
    return [_parse_evaluation(e) for e in evals]


def _bumped_max_tokens(model_args: dict, bumps: int) -> dict:
    """`model_args` with the output budget doubled once per bump."""
    if not bumps or "max_tokens" not in model_args:
        return model_args
    return {**model_args, "max_tokens": model_args["max_tokens"] * (2 ** bumps)}


async def _review_batch(
    texts: List[str], reviewer: Reviewer, semaphore: asyncio.Semaphore,
    limiter: Optional[_RateLimiter] = None,
) -> List[dict]:
    """Send one batch of texts in a single API call. Returns a list of evaluation dicts.

    Four failure kinds are handled differently, because the recovery that fits
    one makes the others worse:

    * **rate limit (429)** — the endpoint would accept this exact request later.
      Wait and re-send it unchanged. Never split: a split answers a refusal to
      serve requests by sending more of them, which is what turned one 429 into
      a cascade of them.
    * **timeout** — the endpoint did not answer at all, usually because the
      request sat in a queue. Re-send it unchanged, and do not split for the
      same reason as above: the halves would aim more requests at something
      already not answering, and each would sit out the whole timeout again.
    * **truncation** — the answer was cut off at ``max_tokens``. Retry with a
      bigger budget. Splitting does not help: ``max_tokens`` is multiplied by
      the batch size in :func:`_prepare_call`, so a batch of 1 gets the same
      budget per article as a batch of 5. (The exception is a config that sets
      no ``max_tokens`` at all: the cap is then the endpoint's, out of reach
      from here, and a shorter answer may fit under it — so that one does split.)
    * **anything else** (wrong item count, malformed JSON) — the model itself is
      struggling with the ask. That is what the split is for.

    This is also where the deferred transport lands its failures: a batched
    request that errored, expired or came back truncated is re-sent through
    here, so it inherits all three recoveries rather than a second one.
    """
    n_articles = len(texts)
    messages, base_args, response_schema = _prepare_call(texts, reviewer)

    # A failed multi-article call is retried once, not max_retries times. Every
    # attempt re-sends all n_articles at full price, and the dominant failure
    # here is a wrong evaluation count — the model silently dropping items —
    # which an identical retry does not fix. The split below is what actually
    # recovers, so go there instead of paying for the same batch twice.
    # Single-article calls keep the full retry budget: for them a retry is cheap
    # and the failure is usually transient.
    attempts = reviewer.max_retries if n_articles == 1 else 1

    last_exc = None
    attempt = 0
    waits = 0       # consecutive 429s; none of these three counters consume
    bumps = 0       # an attempt — each retries a call that never produced
    timeouts = 0    # an answer to judge
    while attempt < attempts:
        model_args = _bumped_max_tokens(base_args, bumps)
        try:
            # The quota slot is taken *inside* the semaphore, immediately
            # before the call. Taken outside it, a task that then queues on a
            # full semaphore stamps the window at a moment it sends nothing,
            # and its real request goes out later — landing in the next window
            # on top of the calls that window has already budgeted for. That is
            # a 429 caused by the pacing itself, and it appears precisely when
            # max_concurrent_requests is the binding constraint.
            async with semaphore:
                if limiter is not None:
                    await limiter.acquire()
                raw, _ = await reviewer.provider_client.call(
                    messages, reviewer.model_id, model_args, response_schema,
                )
            return _evaluations_from_raw(raw, n_articles)

        except Exception as exc:
            last_exc = exc

            if _is_rate_limited(exc) and waits < RATE_LIMIT_MAX_WAITS:
                delay = _rate_limit_delay(exc, waits)
                waits += 1
                # Hold back this reviewer's other in-flight calls too: they share
                # the quota, so letting them through now only earns more 429s.
                if limiter is not None:
                    await limiter.penalise(delay)
                print(f"[{reviewer.name}] rate-limited ({exc}); waiting "
                      f"{delay:.0f}s and re-sending the same batch of "
                      f"{n_articles} ({waits}/{RATE_LIMIT_MAX_WAITS})")
                await asyncio.sleep(delay)
                continue

            if _is_timeout(exc) and timeouts < TIMEOUT_MAX_RETRIES:
                timeouts += 1
                # No backoff: the timeout was itself the wait, and the limiter
                # decides when this re-send may actually leave.
                print(f"[{reviewer.name}] {exc} re-sending the same batch of "
                      f"{n_articles} ({timeouts}/{TIMEOUT_MAX_RETRIES})")
                continue

            if isinstance(exc, TruncatedResponse) and bumps < TRUNCATION_MAX_BUMPS \
                    and "max_tokens" in base_args:
                bumps += 1
                print(f"[{reviewer.name}] {exc}; retrying the batch of "
                      f"{n_articles} with max_tokens="
                      f"{_bumped_max_tokens(base_args, bumps)['max_tokens']}. "
                      f"Raise max_tokens in the config to avoid paying for this "
                      f"twice on every batch.")
                continue

            attempt += 1
            print(f"[{reviewer.name}] attempt {attempt}/{attempts}: {exc}")

    # Attempts exhausted. A multi-article batch commonly fails because the model
    # returns the wrong number of evaluations at large batch sizes (it silently
    # drops items). Rather than crash the whole review, split the batch in two
    # and review each half recursively: this both realigns the counts and
    # shrinks the ask, converging on the single-item schema (which is reliable).
    # A single article that still fails is a genuine error and is raised.
    #
    # Three failures are not of that kind and must not be split:
    #
    # * a rate limit — the quota is the problem, not the batch, and halving it
    #   doubles the number of requests aimed at an endpoint already refusing them;
    # * a timeout — same argument: the endpoint did not answer, and two requests
    #   are not a better way to ask than one. Each half would also wait out the
    #   full timeout before failing, so the split costs twice the delay too;
    # * a truncation we could have budgeted for — halving the batch halves
    #   max_tokens with it, so each half re-truncates at the same point, after
    #   paying its own way back up the doubling ladder.
    #
    # A truncation with no max_tokens in the config *is* splittable: the cap is
    # the endpoint's own, nothing here can raise it, and a shorter answer may fit
    # under it.
    truncated_on_our_budget = (isinstance(last_exc, TruncatedResponse)
                               and "max_tokens" in base_args)
    if n_articles > 1 and not _is_rate_limited(last_exc) \
            and not _is_timeout(last_exc) and not truncated_on_our_budget:
        mid = n_articles // 2
        print(f"[{reviewer.name}] batch of {n_articles} failed ({last_exc}); "
              f"splitting into {mid}+{n_articles - mid} and retrying")
        left = await _review_batch(texts[:mid], reviewer, semaphore, limiter)
        right = await _review_batch(texts[mid:], reviewer, semaphore, limiter)
        return left + right

    if isinstance(last_exc, TruncatedResponse):
        per_article = base_args.get("max_tokens", 0) // max(1, n_articles)
        raise RuntimeError(
            f"[{reviewer.name}] the model kept running out of output budget "
            f"({last_exc}) even after {TRUNCATION_MAX_BUMPS} doublings. Raise "
            f"max_tokens for this reviewer: it is currently {per_article} per "
            f"article (multiplied by items_per_call for the batch), and the "
            f"model needs more than {2 ** TRUNCATION_MAX_BUMPS}x that to answer."
        ) from last_exc

    if _is_rate_limited(last_exc):
        raise RuntimeError(
            f"[{reviewer.name}] still rate-limited after {RATE_LIMIT_MAX_WAITS} "
            f"waits ({last_exc}). Lower requests_per_minute (or "
            f"max_concurrent_requests) for this reviewer."
        ) from last_exc

    if _is_timeout(last_exc):
        raise RuntimeError(
            f"[{reviewer.name}] the endpoint did not answer a batch of "
            f"{n_articles} after {TIMEOUT_MAX_RETRIES} re-sends ({last_exc}). "
            f"Either it is saturated — lower max_concurrent_requests so fewer "
            f"requests queue behind each other — or the batch genuinely needs "
            f"longer than the client allows, in which case lower items_per_call."
        ) from last_exc

    raise RuntimeError(
        f"[{reviewer.name}] failed after {attempts} attempt(s): {last_exc}"
    )


def _plan_calls(texts: List[str], reviewer: Reviewer) -> List[List[str]]:
    """Cut a reviewer's workload into calls of ``items_per_call`` articles."""
    n_per_call = max(1, reviewer.items_per_call)
    return [texts[i: i + n_per_call] for i in range(0, len(texts), n_per_call)]


def _rate_limiter_for(reviewer: Reviewer) -> Optional['_RateLimiter']:
    """The limiter for this reviewer's quota, or None when nothing is paced."""
    return _limiter_for_quota(reviewer.provider, reviewer.host,
                              reviewer.requests_per_minute)


async def _review_all(texts: List[str], reviewer: Reviewer) -> List[dict]:
    """Split texts into batches of items_per_call and run them concurrently."""
    semaphore = asyncio.Semaphore(reviewer.max_concurrent_requests)
    # One limiter per quota, shared with every other reviewer on the same key
    # and kept across chunks: every call drawn on it — the initial ones, their
    # retries, the halves a split produces — is counted once, wherever it starts.
    limiter = _rate_limiter_for(reviewer)
    calls = _plan_calls(texts, reviewer)

    async def _indexed(idx, chunk):
        return idx, await _review_batch(chunk, reviewer, semaphore, limiter)

    ordered = [None] * len(calls)
    tasks = [_indexed(i, c) for i, c in enumerate(calls)]
    async for completed_task in tqdm(asyncio.as_completed(tasks), total=len(calls),
                                     desc=f"[{reviewer.name}] {len(texts)} articles",
                                     unit="batch"):
        idx, result = await completed_task
        ordered[idx] = result

    return [item for chunk in ordered for item in chunk]
