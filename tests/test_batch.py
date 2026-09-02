"""Tests for the provider-agnostic deferred-batch orchestration.

The dangerous failure here is not a crash — it is paying twice. A run that dies
while waiting must re-attach to the batch it already submitted, and must *not*
re-attach to one that was submitted for different work.
"""

import asyncio
import json

import pytest

from pysyrev.core import batch


class _FakeBackend(batch.BatchBackend):
    """Records what it was asked to do; answers every request successfully."""

    name = 'fake'
    max_requests = 100
    max_bytes = 10_000

    def __init__(self, polls_before_end=1, missing=()):
        self.submitted = []          # list of lists of BatchRequest
        self.polled = 0
        self.fetched = []
        self._polls_before_end = polls_before_end
        self._missing = set(missing)  # batch ids that no longer exist

    async def submit(self, requests):
        self.submitted.append(list(requests))
        return f"batch-{len(self.submitted)}"

    async def poll(self, batch_id):
        if batch_id in self._missing:
            raise RuntimeError(f"{batch_id} not found")
        self.polled += 1
        ended = self.polled >= self._polls_before_end
        n = sum(len(chunk) for chunk in self.submitted) or 1
        return batch.BatchStatus(ended=ended, done=n if ended else 0)

    async def fetch(self, batch_id, requests):
        self.fetched.append(batch_id)
        return [batch.BatchResult(rid, content={'echo': req.payload})
                for rid, req in requests.items()]


def _requests(n=3, marker='x'):
    return [batch.BatchRequest(batch.request_id(i), {'body': f"{marker}{i}"})
            for i in range(n)]


class TestRequestId:

    def test_ids_match_the_provider_charset(self):
        # Anthropic accepts [a-zA-Z0-9_-]{1,64}; reviewer names like
        # "Reviewer#1" do not, which is why ids are positional.
        import re
        for i in (0, 7, 999999):
            assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', batch.request_id(i))

    def test_ids_are_unique_and_ordered(self):
        ids = [batch.request_id(i) for i in range(100)]
        assert len(set(ids)) == 100
        assert ids == sorted(ids)


class TestPack:

    def test_a_small_submission_is_one_batch(self):
        assert len(batch.pack(_requests(5), _FakeBackend())) == 1

    def test_the_request_ceiling_splits_the_submission(self):
        backend = _FakeBackend()
        backend.max_requests = 2
        chunks = batch.pack(_requests(5), backend)
        assert [len(c) for c in chunks] == [2, 2, 1]

    def test_the_byte_ceiling_splits_the_submission(self):
        backend = _FakeBackend()
        backend.max_bytes = 40           # a couple of small payloads
        chunks = batch.pack(_requests(6, marker='y' * 10), backend)
        assert len(chunks) > 1
        for chunk in chunks:
            size = sum(len(json.dumps(r.payload).encode()) for r in chunk)
            assert size <= backend.max_bytes or len(chunk) == 1

    def test_a_single_oversized_request_is_not_dropped(self):
        backend = _FakeBackend()
        backend.max_bytes = 1
        chunks = batch.pack(_requests(2), backend)
        assert sum(len(c) for c in chunks) == 2

    def test_no_requests_means_no_batches(self):
        assert batch.pack([], _FakeBackend()) == []


class TestFingerprint:

    def test_the_same_requests_give_the_same_digest(self):
        assert batch.fingerprint(_requests()) == batch.fingerprint(_requests())

    def test_a_changed_payload_changes_the_digest(self):
        assert batch.fingerprint(_requests(marker='a')) != \
               batch.fingerprint(_requests(marker='b'))

    def test_a_changed_count_changes_the_digest(self):
        assert batch.fingerprint(_requests(3)) != batch.fingerprint(_requests(4))

    def test_key_order_inside_a_payload_does_not_change_the_digest(self):
        one = [batch.BatchRequest('call-000000', {'a': 1, 'b': 2})]
        two = [batch.BatchRequest('call-000000', {'b': 2, 'a': 1})]
        assert batch.fingerprint(one) == batch.fingerprint(two)


class TestBatchState:

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / 'sub' / 'state.json')
        batch.BatchState('abc', ['batch-1', 'batch-2'], 123.0).save(path)
        loaded = batch.BatchState.load(path)
        assert loaded.fingerprint == 'abc'
        assert loaded.batch_ids == ['batch-1', 'batch-2']
        assert loaded.submitted_at == 123.0

    def test_a_missing_file_is_no_state(self, tmp_path):
        assert batch.BatchState.load(str(tmp_path / 'nope.json')) is None
        assert batch.BatchState.load(None) is None

    def test_a_corrupt_file_is_no_state_rather_than_a_crash(self, tmp_path):
        path = tmp_path / 'state.json'
        path.write_text('{not json')
        assert batch.BatchState.load(str(path)) is None

    def test_a_file_missing_the_ids_is_no_state(self, tmp_path):
        path = tmp_path / 'state.json'
        path.write_text('{"fingerprint": "abc"}')
        assert batch.BatchState.load(str(path)) is None

    def test_saving_without_a_path_is_a_no_op(self):
        batch.BatchState('abc', ['b']).save(None)   # must not raise


def _run(coro):
    """The project has no pytest-asyncio; drive coroutines the way
    :mod:`pysyrev.core.llm` does at its own entry points."""
    return asyncio.run(coro)


class TestRun:

    def test_results_come_back_keyed_by_request_id(self):
        backend = _FakeBackend()
        results = _run(batch.run(backend, _requests(3), label='r',
                                 poll_interval=0))
        assert set(results) == {batch.request_id(i) for i in range(3)}
        assert results[batch.request_id(1)].content == {'echo': {'body': 'x1'}}
        assert all(r.ok for r in results.values())

    def test_nothing_to_do_submits_nothing(self):
        backend = _FakeBackend()
        assert _run(batch.run(backend, [], label='r')) == {}
        assert backend.submitted == []

    def test_it_waits_until_every_batch_has_ended(self):
        backend = _FakeBackend(polls_before_end=3)
        _run(batch.run(backend, _requests(2), label='r', poll_interval=0))
        assert backend.polled >= 3

    def test_the_batch_ids_are_written_before_the_wait(self, tmp_path):
        state_file = str(tmp_path / 'state.json')
        backend = _FakeBackend()
        _run(batch.run(backend, _requests(2), label='r',
                       state_file=state_file, poll_interval=0))
        saved = batch.BatchState.load(state_file)
        assert saved.batch_ids == ['batch-1']
        assert saved.fingerprint == batch.fingerprint(_requests(2))
        assert saved.submitted_at > 0

    def test_a_rerun_reattaches_instead_of_paying_twice(self, tmp_path):
        state_file = str(tmp_path / 'state.json')
        first = _FakeBackend()
        _run(batch.run(first, _requests(2), label='r',
                       state_file=state_file, poll_interval=0))

        second = _FakeBackend()
        results = _run(batch.run(second, _requests(2), label='r',
                                 state_file=state_file, poll_interval=0))
        assert second.submitted == []                  # nothing resubmitted
        assert second.fetched == ['batch-1']           # the saved batch is read
        assert len(results) == 2

    def test_different_work_does_not_reuse_the_saved_batch(self, tmp_path):
        state_file = str(tmp_path / 'state.json')
        first = _FakeBackend()
        _run(batch.run(first, _requests(2, marker='old'), label='r',
                       state_file=state_file, poll_interval=0))

        second = _FakeBackend()
        _run(batch.run(second, _requests(2, marker='new'), label='r',
                       state_file=state_file, poll_interval=0))
        assert len(second.submitted) == 1              # resubmitted
        assert batch.BatchState.load(state_file).fingerprint == \
               batch.fingerprint(_requests(2, marker='new'))

    def test_a_vanished_batch_is_resubmitted_rather_than_returning_nothing(
            self, tmp_path):
        state_file = str(tmp_path / 'state.json')
        batch.BatchState(batch.fingerprint(_requests(2)),
                         ['batch-gone'], 1.0).save(state_file)
        backend = _FakeBackend(missing=['batch-gone'])
        results = _run(batch.run(backend, _requests(2), label='r',
                                 state_file=state_file, poll_interval=0))
        assert len(backend.submitted) == 1
        assert len(results) == 2

    def test_running_without_a_state_file_still_works(self):
        backend = _FakeBackend()
        results = _run(batch.run(backend, _requests(1), label='r',
                                 poll_interval=0))
        assert len(results) == 1

    def test_waiting_past_the_deadline_raises_and_keeps_the_ids(self, tmp_path):
        state_file = str(tmp_path / 'state.json')
        backend = _FakeBackend(polls_before_end=10 ** 6)
        with pytest.raises(TimeoutError, match='re-attach'):
            _run(batch.run(backend, _requests(2), label='r',
                           state_file=state_file, poll_interval=0, max_wait=-1))
        # The point of that error message: the ids survive the failure.
        assert batch.BatchState.load(state_file).batch_ids == ['batch-1']

    def test_a_multi_batch_submission_is_collected_as_one(self):
        backend = _FakeBackend()
        backend.max_requests = 2
        results = _run(batch.run(backend, _requests(5), label='r',
                                 poll_interval=0))
        assert len(backend.submitted) == 3
        assert len(results) == 5
