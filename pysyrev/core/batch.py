"""Deferred batch execution — provider-agnostic.

Several LLM vendors expose the same bargain: send a pile of requests you do not
need answered *now*, get them back within a day, pay half. Anthropic calls it
the Message Batches API, OpenAI calls it the Batch API; both take the requests
their synchronous endpoint takes, key them by a caller-chosen id, and hand the
answers back in arbitrary order.

This module owns everything that bargain implies *regardless of vendor*:

* **Packing.** Limits are per batch, not per run (Anthropic: 100k requests /
  256 MB; OpenAI: 50k / 200 MB), so a corpus is split into as many batches as
  its own size demands and awaited as one.
* **Resumption.** A batch is a checkpoint: the vendor holds the results for
  weeks. A run that dies while waiting must re-attach to the batch it has
  already paid for, never submit a second one. That is what ``state_file``
  buys — the batch ids on disk, alongside a fingerprint of the exact requests
  they carry. A matching fingerprint means those batches *are* this run's work;
  a mismatch (corpus re-cleaned, prompt edited) means they are not, and the
  state is discarded rather than trusted.
* **Waiting.** Progress is reported per request, not per batch, and the wait is
  bounded — the vendors' only guarantee is 24 hours.

What it does *not* own is any knowledge of a provider: submitting, polling and
decoding are three methods on :class:`BatchBackend`, implemented next to the
provider they belong to (see ``_AnthropicBatchBackend`` in
:mod:`pysyrev.core.llm`). A backend hands back :class:`BatchResult` objects
whose ``content`` has already been extracted by the same code the synchronous
path uses, so nothing vendor-shaped crosses back into this module.

Latency is not throughput: this is the right transport for an overnight
screening run and the wrong one for anything interactive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm.auto import tqdm

#: Give up waiting past the vendors' own 24 h deadline, after which unfinished
#: requests come back marked expired.
MAX_WAIT_SECONDS: float = 25 * 3600

DEFAULT_POLL_INTERVAL: float = 30.0


# ── The unit of work ───────────────────────────────────────────────────────

@dataclass
class BatchRequest:
    """One deferred call: a caller-chosen id and a provider-native payload."""
    custom_id: str
    payload: dict


@dataclass
class BatchStatus:
    """Where one submitted batch stands."""
    ended: bool
    done: int           # requests finished, whatever their outcome


@dataclass
class BatchResult:
    """The outcome of one deferred call.

    ``content`` is what the provider's synchronous path would have returned for
    the same request — already extracted by the backend, so callers parse batch
    and non-batch answers with the same code. ``error`` is set instead when the
    request failed, expired, or was cancelled.
    """
    custom_id: str
    content: object = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def request_id(index: int) -> str:
    """Id for the *index*-th request of a submission.

    Vendors constrain the id charset (Anthropic: ``[a-zA-Z0-9_-]{1,64}``),
    which reviewer names do not satisfy — ``Reviewer#1`` is rejected. A
    positional id sidesteps the question; the caller keeps the mapping from
    index to work, which it can rebuild deterministically anyway.
    """
    return f"call-{index:06d}"


# ── Backend interface ──────────────────────────────────────────────────────

class BatchBackend:
    """What a provider must supply for its deferred-batch endpoint to be usable."""

    #: Shown in log lines.
    name: str = 'batch'
    #: Per-batch ceilings; packing respects both.
    max_requests: int = 50_000
    max_bytes: int = 200 * 1024 * 1024

    async def submit(self, requests: List[BatchRequest]) -> str:
        """Create one batch and return its provider-side id."""
        raise NotImplementedError

    async def poll(self, batch_id: str) -> BatchStatus:
        """Current state of a submitted batch."""
        raise NotImplementedError

    async def fetch(self, batch_id: str,
                    requests: Dict[str, BatchRequest]) -> List[BatchResult]:
        """Results of an ended batch, in whatever order the provider returns them.

        *requests* maps every id in the batch to what was sent for it, so the
        backend can decode a response the same way the synchronous path would —
        which depends on the request (a forced-tool call and a free-form one do
        not carry their answer in the same place).
        """
        raise NotImplementedError

    async def exists(self, batch_id: str) -> bool:
        """Whether a previously submitted batch is still retrievable."""
        try:
            await self.poll(batch_id)
            return True
        except Exception:
            return False


# ── Packing ────────────────────────────────────────────────────────────────

def pack(requests: List[BatchRequest], backend: BatchBackend) -> List[List[BatchRequest]]:
    """Split *requests* into submittable batches, by count and by payload size."""
    batches: List[List[BatchRequest]] = []
    current: List[BatchRequest] = []
    current_bytes = 0
    for request in requests:
        size = len(json.dumps(request.payload, ensure_ascii=False).encode())
        too_many = len(current) >= backend.max_requests
        too_big = bool(current) and current_bytes + size > backend.max_bytes
        if too_many or too_big:
            batches.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def fingerprint(requests: List[BatchRequest]) -> str:
    """Stable digest of a submission, used to validate a resumed batch.

    Covers the whole payload — prompts, model, sampling parameters, article
    text — so any change to what the model would see invalidates the saved ids
    instead of quietly reusing results computed for something else.
    """
    digest = hashlib.sha256()
    for request in requests:
        digest.update(request.custom_id.encode())
        digest.update(json.dumps(request.payload, sort_keys=True,
                                 ensure_ascii=False).encode())
    return digest.hexdigest()


# ── Persisted state ────────────────────────────────────────────────────────

@dataclass
class BatchState:
    """The batch ids a submission is waiting on, and what they were sent for."""
    fingerprint: str
    batch_ids: List[str] = field(default_factory=list)
    submitted_at: float = 0.0

    @classmethod
    def load(cls, path: Optional[str]) -> Optional['BatchState']:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as handle:
                data = json.load(handle)
            return cls(fingerprint=data['fingerprint'],
                       batch_ids=list(data['batch_ids']),
                       submitted_at=float(data.get('submitted_at', 0.0)))
        except (OSError, ValueError, KeyError, TypeError):
            return None                     # unreadable state is no state

    def save(self, path: Optional[str]) -> None:
        if not path:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump({'fingerprint': self.fingerprint,
                       'batch_ids': self.batch_ids,
                       'submitted_at': self.submitted_at}, handle, indent=2)


# ── Orchestration ──────────────────────────────────────────────────────────

async def _reattach(backend: BatchBackend, state: BatchState,
                    label: str) -> Optional[List[str]]:
    """Confirm every saved id is still retrievable, or give up on the state.

    Beyond the provider's retention window — or with ids from another account —
    the only honest move is to resubmit and pay again, rather than return
    nothing and let the caller believe the round was reviewed.
    """
    for batch_id in state.batch_ids:
        if not await backend.exists(batch_id):
            print(f"[{backend.name}] {label}: saved batch {batch_id} is no "
                  f"longer retrievable; resubmitting.")
            return None
    waited = (time.time() - state.submitted_at) / 60 if state.submitted_at else 0
    print(f"[{backend.name}] {label}: re-attaching to {len(state.batch_ids)} "
          f"batch(es) submitted {waited:.0f} min ago — not resubmitting.")
    return list(state.batch_ids)


async def _submit_all(backend: BatchBackend, chunks: List[List[BatchRequest]],
                      label: str) -> List[str]:
    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch_id = await backend.submit(chunk)
        batch_ids.append(batch_id)
        suffix = f" ({i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        print(f"[{backend.name}] {label}: submitted {len(chunk)} request(s) "
              f"as {batch_id}{suffix}")
    return batch_ids


async def _wait(backend: BatchBackend, batch_ids: List[str], n_requests: int,
                label: str, poll_interval: float, max_wait: float) -> None:
    """Poll until every batch has ended, reporting progress per request."""
    deadline = time.time() + max_wait
    bar = tqdm(total=n_requests, desc=f"[{backend.name}] {label}", unit="req")
    done_before = 0
    try:
        while True:
            done, pending = 0, False
            for batch_id in batch_ids:
                status = await backend.poll(batch_id)
                done += status.done
                pending = pending or not status.ended
            bar.update(max(0, done - done_before))
            done_before = done
            if not pending:
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"[{backend.name}] {label}: still processing after "
                    f"{max_wait / 3600:.1f} h ({done}/{n_requests} done). The "
                    f"batch ids are on disk — rerun to re-attach rather than "
                    f"resubmit."
                )
            await asyncio.sleep(poll_interval)
    finally:
        bar.close()


async def run(backend: BatchBackend, requests: List[BatchRequest], *,
              label: str,
              state_file: Optional[str] = None,
              poll_interval: float = DEFAULT_POLL_INTERVAL,
              max_wait: float = MAX_WAIT_SECONDS) -> Dict[str, BatchResult]:
    """Submit *requests*, wait for them, and return ``{custom_id: BatchResult}``.

    Results are keyed by id and never by position: providers return them in
    arbitrary order, and a batch that partially failed returns fewer of them
    than were sent.
    """
    if not requests:
        return {}

    digest = fingerprint(requests)
    state = BatchState.load(state_file)
    batch_ids = None
    if state is not None and state.fingerprint == digest:
        batch_ids = await _reattach(backend, state, label)
    elif state is not None:
        print(f"[{backend.name}] {label}: the saved batch does not match the "
              f"requests about to be sent (corpus or prompt changed); "
              f"submitting a new one.")

    if batch_ids is None:
        batch_ids = await _submit_all(backend, pack(requests, backend), label)
        BatchState(fingerprint=digest, batch_ids=batch_ids,
                   submitted_at=time.time()).save(state_file)

    await _wait(backend, batch_ids, len(requests), label, poll_interval, max_wait)

    by_id = {request.custom_id: request for request in requests}
    results: Dict[str, BatchResult] = {}
    for batch_id in batch_ids:
        for result in await backend.fetch(batch_id, by_id):
            results[result.custom_id] = result
    return results
