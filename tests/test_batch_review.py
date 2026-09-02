"""Tests for reviewing a round through a provider's deferred-batch endpoint.

Two properties matter beyond "it returns something":

* a batched run must send the *same* request a live one would — the estimate,
  the prompt cache and the results all depend on that;
* evaluations must land on the right rows. Batch results come back keyed and
  unordered, and the pipeline assigns them positionally.
"""

import asyncio
import re

import pandas as pd
import pytest

from pysyrev.core import batch as deferred
from pysyrev.core.llm import (BatchRun, Reviewer, _prepare_call,
                              _review_round_deferred, _run_workflow,
                              check_batch_support, run_review)
from pysyrev.review import _check_batch_api


# ── Doubles ────────────────────────────────────────────────────────────────

class _FakeBackend(deferred.BatchBackend):
    """Answers every request with one evaluation per article it was sent."""

    name = 'fake-batch'

    def __init__(self, fail_ids=(), wrong_count_ids=()):
        self.submitted = []
        self._fail = set(fail_ids)
        self._wrong = set(wrong_count_ids)

    async def submit(self, requests):
        self.submitted.extend(requests)
        return f"batch-{len(self.submitted)}"

    async def poll(self, batch_id):
        return deferred.BatchStatus(ended=True, done=len(self.submitted))

    async def fetch(self, batch_id, requests):
        out = []
        for rid, request in requests.items():
            if rid in self._fail:
                out.append(deferred.BatchResult(rid, error='overloaded_error'))
                continue
            n = len(request.payload['texts'])
            if rid in self._wrong:
                n -= 1                      # the model silently dropped one
            out.append(deferred.BatchResult(rid, content={'evaluations': [
                {'evaluation': 4, 'reasoning': f"batched:{t}"}
                for t in request.payload['texts'][:n]
            ]}))
        return out


def _articles(user_prompt: str) -> list:
    """The articles a real ``_user_prompt`` carries, recovered from the prompt.

    Reading them back out — rather than having the double remember what it was
    handed — keeps the fakes honest: they see exactly what a provider sees.
    """
    blocks = re.split(r'--- Article \d+ ---\n', user_prompt)
    if len(blocks) > 1:
        return [b.strip() for b in blocks[1:]]
    return [user_prompt.strip()]


class _FakeProvider:
    """A provider whose payload is legible enough to assert on."""

    def __init__(self, backend=None):
        self._backend = backend
        self.sync_calls = []

    def batch_backend(self):
        return self._backend

    def build_payload(self, messages, model_id, model_args, response_schema):
        return {'model': model_id, 'system': messages[0]['content'],
                'user': messages[1]['content'], 'args': dict(model_args),
                'schema': response_schema.__name__,
                # not part of a real payload — lets the fake backend answer
                'texts': _articles(messages[1]['content'])}

    async def call(self, messages, model_id, model_args, response_schema):
        texts = _articles(messages[1]['content'])
        self.sync_calls.append(texts)
        return {'evaluations': [{'evaluation': 2, 'reasoning': f"sync:{t}"}
                                for t in texts]}, 0.0


class _NoBatchProvider(_FakeProvider):
    def batch_backend(self):
        return None


def _reviewer(name, provider, items_per_call=2):
    return Reviewer(
        name=name, model_id='claude-haiku-4-5', provider='fake',
        provider_client=provider, backstory='bs', reasoning=None,
        max_tokens=200, temperature=0.2, reasoning_effort=None,
        additional_context=None, inclusion_criteria='inc',
        exclusion_criteria='exc', input_description='article',
        max_retries=2, max_concurrent_requests=2,
        items_per_call=items_per_call,
    )


def _run(coro):
    return asyncio.run(coro)


# ── The requests actually sent ─────────────────────────────────────────────

class TestRequestParity:
    """The deferred transport must send what the synchronous one sends."""

    def test_the_payload_is_built_from_the_same_prepared_call(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key-not-used')
        from pysyrev.core.llm import _AnthropicProvider, _ReviewBatch
        provider = _AnthropicProvider()
        reviewer = _reviewer('Reviewer#1', provider, items_per_call=3)

        messages, model_args, schema = _prepare_call(['a', 'b', 'c'], reviewer)
        payload = provider.build_payload(messages, reviewer.model_id,
                                         model_args, schema)

        assert payload['model'] == 'claude-haiku-4-5'
        assert payload['system'] == messages[0]['content']
        assert payload['messages'] == [messages[1]]
        # Structured output goes through a forced tool — the tool schema and
        # the tool-use preamble are real tokens the estimate counts.
        assert payload['tool_choice'] == {'type': 'tool', 'name': 'review'}
        assert payload['tools'][0]['input_schema'] == _ReviewBatch.model_json_schema()
        # max_tokens is per article and must be scaled for a multi-item call.
        assert payload['max_tokens'] == 200 * 3
        assert payload['temperature'] == 0.2

    def test_reasoning_effort_is_dropped_from_the_payload_as_in_a_live_call(
            self, monkeypatch, capsys):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key-not-used')
        from pysyrev.core.llm import _AnthropicProvider
        _AnthropicProvider._warned_unsupported = set()
        provider = _AnthropicProvider()
        reviewer = _reviewer('Reviewer#3', provider, items_per_call=1)
        reviewer.reasoning_effort = 'medium'

        messages, model_args, schema = _prepare_call(['a'], reviewer)
        payload = provider.build_payload(messages, reviewer.model_id,
                                         model_args, schema)
        assert 'reasoning_effort' not in payload
        assert 'reasoning_effort' in capsys.readouterr().out

    def test_one_request_is_planned_per_items_per_call_chunk(self):
        backend = _FakeBackend()
        reviewer = _reviewer('Reviewer#1', _FakeProvider(backend),
                             items_per_call=2)
        _run(_review_round_deferred([reviewer], ['t1', 't2', 't3', 't4', 't5'],
                                    'A', BatchRun(poll_interval=0)))
        assert len(backend.submitted) == 3          # 2 + 2 + 1


# ── Results landing on the right rows ──────────────────────────────────────

class TestRoundResults:

    def test_evaluations_come_back_in_text_order(self):
        reviewer = _reviewer('Reviewer#1', _FakeProvider(_FakeBackend()))
        out = _run(_review_round_deferred(
            [reviewer], ['t1', 't2', 't3'], 'A', BatchRun(poll_interval=0)))
        assert [e['reasoning'] for e in out['Reviewer#1']] == \
               ['batched:t1', 'batched:t2', 'batched:t3']

    def test_every_reviewer_of_a_round_shares_one_submission(self):
        backend = _FakeBackend()
        provider = _FakeProvider(backend)
        reviewers = [_reviewer('Reviewer#1', provider),
                     _reviewer('Reviewer#2', provider)]
        out = _run(_review_round_deferred(reviewers, ['t1', 't2'], 'A',
                                          BatchRun(poll_interval=0)))
        # One wait, not one per reviewer: reviewers of a round are independent.
        assert len(backend.submitted) == 2          # one call each
        assert set(out) == {'Reviewer#1', 'Reviewer#2'}
        assert len(out['Reviewer#1']) == len(out['Reviewer#2']) == 2

    def test_reviewers_on_different_clients_get_their_own_submission(self):
        first, second = _FakeBackend(), _FakeBackend()
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(first)),
                     _reviewer('Reviewer#2', _FakeProvider(second))]
        out = _run(_review_round_deferred(reviewers, ['t1', 't2'], 'A',
                                          BatchRun(poll_interval=0)))
        # A batch can only be polled at the endpoint it was submitted to.
        assert len(first.submitted) == 1 and len(second.submitted) == 1
        assert len(out['Reviewer#1']) == len(out['Reviewer#2']) == 2


class TestFailureFallback:

    def test_a_failed_request_is_re_reviewed_synchronously(self):
        backend = _FakeBackend(fail_ids={deferred.request_id(0)})
        provider = _FakeProvider(backend)
        reviewer = _reviewer('Reviewer#1', provider, items_per_call=2)
        out = _run(_review_round_deferred(
            [reviewer], ['t1', 't2', 't3', 't4'], 'A', BatchRun(poll_interval=0)))

        # First call recovered live, second served from the batch — and the
        # articles stay in order across the two transports.
        assert [e['reasoning'] for e in out['Reviewer#1']] == \
               ['sync:t1', 'sync:t2', 'batched:t3', 'batched:t4']
        # Exactly the failed chunk was resent — not the whole round.
        assert provider.sync_calls == [['t1', 't2']]

    def test_a_short_answer_is_caught_and_re_reviewed(self):
        # The dominant real failure: the model silently drops articles, so the
        # count no longer matches the rows the caller will assign to.
        backend = _FakeBackend(wrong_count_ids={deferred.request_id(0)})
        provider = _FakeProvider(backend)
        reviewer = _reviewer('Reviewer#1', provider, items_per_call=2)
        out = _run(_review_round_deferred(
            [reviewer], ['t1', 't2'], 'A', BatchRun(poll_interval=0)))
        assert [e['reasoning'] for e in out['Reviewer#1']] == ['sync:t1', 'sync:t2']

    def test_a_truncated_batched_answer_is_recognised_as_truncated(self):
        """A deferred reply can run out of output budget like a live one.

        It carries the same ``Message``, so the same ``stop_reason`` check has
        to fire — otherwise the cut-off JSON reaches the parser and comes back
        as a delimiter error, and the run splits the batch instead of raising
        the budget, which for a per-article budget fixes nothing.
        """
        from pysyrev.core.llm import _AnthropicProvider, TruncatedResponse

        class _Cutoff:
            stop_reason = 'max_tokens'
            content = []

        with pytest.raises(TruncatedResponse) as excinfo:
            _AnthropicProvider.extract_content(
                _Cutoff(), [{'name': 'review'}], {'max_tokens': 600})
        # The message names the budget that ran out, not a JSON position.
        assert '600' in str(excinfo.value)

    def test_an_untruncated_batched_answer_is_decoded_normally(self):
        from pysyrev.core.llm import _AnthropicProvider

        class _Block:
            type = 'tool_use'
            input = {'evaluations': [{'evaluation': 4, 'reasoning': 'ok'}]}

        class _Message:
            stop_reason = 'tool_use'
            content = [_Block()]

        assert _AnthropicProvider.extract_content(
            _Message(), [{'name': 'review'}], {'max_tokens': 600}
        ) == {'evaluations': [{'evaluation': 4, 'reasoning': 'ok'}]}

    def test_the_fallback_is_announced_with_its_reason(self, capsys):
        backend = _FakeBackend(fail_ids={deferred.request_id(0)})
        reviewer = _reviewer('Reviewer#1', _FakeProvider(backend))
        _run(_review_round_deferred([reviewer], ['t1', 't2'], 'A',
                                    BatchRun(poll_interval=0)))
        out = capsys.readouterr().out
        assert 'full price' in out
        assert 'overloaded_error' in out


# ── Resumption ─────────────────────────────────────────────────────────────

class TestStateFile:

    def test_each_round_remembers_its_own_batch(self, tmp_path):
        run = BatchRun(state_dir=str(tmp_path))
        assert run.state_file('A').endswith('batch_round-A.json')
        assert run.state_file('A') != run.state_file('B')

    def test_no_state_dir_means_no_state_file(self):
        assert BatchRun().state_file('A') is None

    def test_a_second_run_reattaches_to_the_first_submission(self, tmp_path):
        texts = ['t1', 't2', 't3']
        first = _FakeBackend()
        _run(_review_round_deferred(
            [_reviewer('Reviewer#1', _FakeProvider(first))], texts, 'A',
            BatchRun(poll_interval=0, state_dir=str(tmp_path))))

        second = _FakeBackend()
        out = _run(_review_round_deferred(
            [_reviewer('Reviewer#1', _FakeProvider(second))], texts, 'A',
            BatchRun(poll_interval=0, state_dir=str(tmp_path))))
        assert second.submitted == []           # the round was already paid for
        assert len(out['Reviewer#1']) == 3


# ── Wiring ─────────────────────────────────────────────────────────────────

class TestProviderSupport:

    def test_a_provider_without_the_endpoint_is_named(self):
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(_FakeBackend())),
                     _reviewer('Reviewer#2', _NoBatchProvider())]
        assert check_batch_support(reviewers) == ['Reviewer#2']

    def test_all_supported_means_an_empty_list(self):
        backend = _FakeBackend()
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(backend))]
        assert check_batch_support(reviewers) == []


class TestConfigGate:
    """`use_batch_api` must never fail open — a silent fallback doubles the bill."""

    def test_off_stays_off(self, capsys):
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(_FakeBackend()))]
        assert _check_batch_api(reviewers, False) is False
        assert capsys.readouterr().out == ''

    def test_on_with_a_capable_provider_is_silent(self, capsys):
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(_FakeBackend()))]
        assert _check_batch_api(reviewers, True) is True
        assert capsys.readouterr().out == ''

    def test_on_without_the_endpoint_falls_back_loudly(self, capsys):
        reviewers = [_reviewer('Reviewer#1', _FakeProvider(_FakeBackend())),
                     _reviewer('Reviewer#2', _NoBatchProvider())]
        assert _check_batch_api(reviewers, True) is False
        out = capsys.readouterr().out
        assert 'Reviewer#2' in out
        assert 'full price' in out


def _dataset(n=4):
    return pd.DataFrame({'title': [f"t{i}" for i in range(n)],
                         'abstract': [f"a{i}" for i in range(n)]})


def _schema(reviewers):
    return [{'round': 'A', 'reviewers': reviewers,
             'text_inputs': ['title', 'abstract']}]


class TestRunReviewWiring:

    def test_batch_mode_bypasses_checkpoint_chunking(self, capsys):
        backend = _FakeBackend()
        reviewer = _reviewer('Reviewer#1', _FakeProvider(backend),
                             items_per_call=4)
        out = run_review(_dataset(8), _schema([reviewer]), 'majority',
                         batch_size=2, sample_size=None, pause=0,
                         batch_run=BatchRun(poll_interval=0))
        # Chunking would have meant four sequential submissions, each with its
        # own wait, for no saving.
        assert len(backend.submitted) == 2       # 8 articles / 4 per call
        assert len(out) == 8
        assert 'batch_size=2 is ignored' in capsys.readouterr().out

    def test_without_a_batch_run_nothing_is_submitted(self):
        backend = _FakeBackend()
        provider = _FakeProvider(backend)
        reviewer = _reviewer('Reviewer#1', provider, items_per_call=4)
        run_review(_dataset(4), _schema([reviewer]), 'majority',
                   batch_size=None, sample_size=None, pause=0)
        assert backend.submitted == []
        assert provider.sync_calls                # the live path ran instead

    def test_the_workflow_writes_the_same_columns_either_way(self):
        backend = _FakeBackend()
        reviewer = _reviewer('Reviewer#1', _FakeProvider(backend),
                             items_per_call=2)
        batched = _run(_run_workflow(_dataset(4), _schema([reviewer]),
                                     BatchRun(poll_interval=0)))
        live = _run(_run_workflow(_dataset(4), _schema([reviewer])))
        assert list(batched.columns) == list(live.columns)
        assert 'round-A_Reviewer#1_evaluation' in batched.columns
        assert batched['round-A_Reviewer#1_evaluation'].tolist() == [4] * 4

    def test_an_empty_round_submits_nothing(self):
        backend = _FakeBackend()
        reviewer = _reviewer('Reviewer#1', _FakeProvider(backend))
        out = _run(_run_workflow(_dataset(0), _schema([reviewer]),
                                 BatchRun(poll_interval=0)))
        assert backend.submitted == []
        assert len(out) == 0
