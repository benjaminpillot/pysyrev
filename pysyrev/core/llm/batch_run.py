"""
The deferred transport: a whole round handed to the provider's batch endpoint.

Same prompts, same ``items_per_call`` packing, same parsing as the live path —
submitted in bulk, answered within the day, billed at half price. An
interrupted run re-attaches to the batch already paid for rather than
resubmitting it. Calls that come back unusable fall back to the synchronous
path, which knows how to retry and how to split a batch the model mangled.

The vendor-neutral half of this — packing, resumption, waiting — lives in
:mod:`pysyrev.core.batch`.
"""

import asyncio
import os.path
from dataclasses import dataclass
from typing import List, Optional

from pysyrev.core import batch as deferred
from pysyrev.core.llm.reviewer import (Reviewer, _evaluations_from_raw,
                                       _plan_calls, _prepare_call,
                                       _rate_limiter_for, _review_batch)


@dataclass
class BatchRun:
    """Settings for reviewing through a provider's deferred-batch endpoint.

    Passing one of these to :func:`run_review` swaps the transport, not the
    work: the same prompts, the same ``items_per_call`` packing, the same
    parsing — submitted in bulk and collected within the day, at half price.
    """
    poll_interval: float = deferred.DEFAULT_POLL_INTERVAL
    max_wait:      float = deferred.MAX_WAIT_SECONDS
    state_dir:     Optional[str] = None      # where batch ids are remembered

    def state_file(self, round_name) -> Optional[str]:
        if not self.state_dir:
            return None
        return os.path.join(self.state_dir, f"batch_round-{round_name}.json")


async def _review_round_deferred(reviewers: List[Reviewer], texts: List[str],
                                 round_name, run: BatchRun) -> dict:
    """Review one round's reviewers in a single deferred submission.

    The reviewers of a round are independent — they screen the same articles
    and never read each other's scores — so they share one batch and one wait.
    Rounds cannot be merged: round *n+1* only sees what round *n* left
    unsettled, which is not known until *n* comes back.

    Returns ``{reviewer name: [evaluation, ...]}`` aligned with *texts*.
    """
    # One submission per distinct client. Usually there is exactly one, but two
    # reviewers may sit behind different accounts or hosts, and a batch can only
    # be submitted to the endpoint that will be polled for it.
    groups: dict = {}
    for reviewer in reviewers:
        groups.setdefault(id(reviewer.provider_client), []).append(reviewer)

    n_calls = 0
    plan = []                        # plan[i] = (reviewer, that call's articles)
    submissions = []
    for group_idx, group in enumerate(groups.values()):
        client = group[0].provider_client
        group_requests = []
        for reviewer in group:
            for chunk in _plan_calls(texts, reviewer):
                messages, model_args, schema = _prepare_call(chunk, reviewer)
                payload = client.build_payload(messages, reviewer.model_id,
                                               model_args, schema)
                group_requests.append(
                    deferred.BatchRequest(deferred.request_id(n_calls), payload))
                plan.append((reviewer, chunk))
                n_calls += 1
        # A single group is the normal case and keeps the plain round names, so
        # a state file written before a second provider appeared still matches.
        suffix = '' if len(groups) == 1 else f" / {group[0].provider}"
        submissions.append(deferred.run(
            client.batch_backend(), group_requests,
            label=f"round {round_name}{suffix}",
            state_file=run.state_file(round_name if len(groups) == 1
                                      else f"{round_name}-{group_idx}"),
            poll_interval=run.poll_interval,
            max_wait=run.max_wait,
        ))

    results = {}
    for collected in await asyncio.gather(*submissions):
        results.update(collected)

    # A deferred batch has no retry of its own: a request that errored, expired
    # or came back with the wrong number of evaluations is re-reviewed through
    # the synchronous path, which already knows how to retry and how to split a
    # batch the model mangled. That tail is billed at full price — it is meant
    # to stay a tail.
    outcomes: List[Optional[List[dict]]] = [None] * len(plan)
    failed: List[tuple] = []
    for i, (_, chunk) in enumerate(plan):
        result = results.get(deferred.request_id(i))
        if result is None:
            failed.append((i, 'no result returned for this request'))
        elif not result.ok:
            failed.append((i, result.error))
        else:
            try:
                outcomes[i] = _evaluations_from_raw(result.content, len(chunk))
            except Exception as exc:
                failed.append((i, str(exc)))

    if failed:
        reasons = sorted({reason for _, reason in failed})[:3]
        print(f"[batch] round {round_name}: {len(failed)}/{len(plan)} call(s) "
              f"came back unusable — re-reviewing them synchronously at full "
              f"price. Reasons: {'; '.join(reasons)}")
        semaphores = {r.name: asyncio.Semaphore(r.max_concurrent_requests)
                      for r in reviewers}
        limiters = {r.name: _rate_limiter_for(r) for r in reviewers}

        async def _resend(idx):
            reviewer, chunk = plan[idx]
            return idx, await _review_batch(chunk, reviewer,
                                            semaphores[reviewer.name],
                                            limiters[reviewer.name])

        for coro in asyncio.as_completed([_resend(i) for i, _ in failed]):
            idx, evaluations = await coro
            outcomes[idx] = evaluations

    by_reviewer = {reviewer.name: [] for reviewer in reviewers}
    for (reviewer, _), evaluations in zip(plan, outcomes):
        by_reviewer[reviewer.name].extend(evaluations)
    return by_reviewer


def check_batch_support(reviewers: List[Reviewer]) -> List[str]:
    """Names of reviewers whose provider has no deferred-batch endpoint."""
    return [r.name for r in reviewers if r.provider_client.batch_backend() is None]
