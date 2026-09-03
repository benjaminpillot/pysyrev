"""Tests for the Albert (Etalab / DINUM) provider.

Albert speaks OpenAI's transport but serves open-weight models through vLLM, so
what a deployment does with `response_format` is not knowable from the config.
The provider therefore starts strict (`json_schema`), degrades on refusal, and
sticks to the level that worked. These tests pin that ladder, the credential
handling, and the parameter filtering — none of them touch the network.
"""

import asyncio
import json

import pytest

from pysyrev.core.llm import (ALBERT_API_KEY_ENV, ALBERT_DEFAULT_HOST,
                              _AlbertProvider, _make_provider, _ReviewItem)

pytest.importorskip("openai")


# ── Fake OpenAI client ─────────────────────────────────────────────────────

class _BadRequest(Exception):
    """Stand-in for openai.BadRequestError (HTTP 400)."""
    status_code = 400


class _FakeCompletions:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _fake_response(reply)


class _FakeClient:
    def __init__(self, *replies):
        self.completions = _FakeCompletions(replies)
        self.chat = self

    @property
    def calls(self):
        return self.completions.calls


def _fake_response(content):
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    return _Resp(content)


def _provider(monkeypatch, *replies):
    monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
    provider = _AlbertProvider()
    provider._client = _FakeClient(*replies)
    return provider


def _call(provider, schema=_ReviewItem, model_args=None):
    return asyncio.run(provider.call(
        [{"role": "system", "content": "sys"},
         {"role": "user", "content": "article"}],
        "albert-large", model_args or {}, schema,
    ))


_ITEM = '{"evaluation": 4, "reasoning": "on topic"}'


# ── Credentials and endpoint ───────────────────────────────────────────────

class TestCredentials:

    def test_key_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        provider = _AlbertProvider()
        assert provider._client.api_key == "test-key"

    def test_defaults_to_the_public_endpoint(self, monkeypatch):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        assert str(_AlbertProvider()._client.base_url).rstrip("/") == ALBERT_DEFAULT_HOST

    def test_host_overrides_the_default_endpoint(self, monkeypatch):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        provider = _AlbertProvider(host="https://albert.internal/v1")
        assert str(provider._client.base_url).rstrip("/") == "https://albert.internal/v1"

    def test_missing_key_names_the_variable_to_set(self, monkeypatch):
        monkeypatch.delenv(ALBERT_API_KEY_ENV, raising=False)
        with pytest.raises(ValueError, match=ALBERT_API_KEY_ENV):
            _AlbertProvider()

    def test_registered_under_albert(self, monkeypatch):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        assert isinstance(_make_provider("albert", None), _AlbertProvider)


class TestTransportRetries:
    """The SDK must not retry: it does not know about the quota."""

    def test_the_sdk_does_not_retry_behind_the_limiter(self, monkeypatch):
        """Its default is two silent retries on a 429, so one metered call
        would cost three requests against a ten-a-minute quota — and the 429
        that finally surfaces would already have burnt the other two."""
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        assert _AlbertProvider()._client.max_retries == 0

    def test_a_plain_openai_client_keeps_the_sdk_default(self):
        """Nothing meters that one, so the SDK's backoff is the only recovery."""
        from pysyrev.core.llm import _OpenAIProvider
        assert _OpenAIProvider(api_key="k")._client.max_retries == 2

    def test_a_call_gives_up_long_before_the_sdk_would(self, monkeypatch):
        """600s is not a timeout so much as an absence of one: the request holds
        a concurrency slot for ten minutes before anyone hears about it."""
        from pysyrev.core.llm import ALBERT_TIMEOUT
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        assert _AlbertProvider()._client.timeout == ALBERT_TIMEOUT
        assert ALBERT_TIMEOUT < 600

    def test_the_config_overrides_the_albert_default(self, monkeypatch):
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        assert _AlbertProvider(timeout=45.0)._client.timeout == 45.0

    def test_every_provider_accepts_one(self, monkeypatch):
        """Not just Albert: a hung request holds a concurrency slot everywhere,
        and litellm's own default is 6000s."""
        from pysyrev.core.llm import _make_provider, _LiteLLMProvider
        monkeypatch.setenv(ALBERT_API_KEY_ENV, "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert _make_provider("albert", None, 45.0)._client.timeout == 45.0
        assert _make_provider("openai", None, 45.0)._client.timeout == 45.0
        assert _make_provider("ollama", None, 45.0)._client.timeout == 45.0
        assert _make_provider("anthropic", None, 45.0)._client.timeout == 45.0
        litellm_provider = _make_provider("litellm", None, 45.0)
        assert isinstance(litellm_provider, _LiteLLMProvider)
        assert litellm_provider._timeout == 45.0

    def test_a_billed_provider_is_left_on_its_own_default(self, monkeypatch):
        """Cutting a request that would have finished bills the tokens and
        returns nothing, so that trade-off is the config's to make."""
        from pysyrev.core.llm import _make_provider, _OpenAIProvider
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert _OpenAIProvider(api_key="k")._client.timeout.read == 600
        assert _make_provider("anthropic", None)._client.timeout.read == 600

    def test_a_demotion_takes_a_quota_slot_of_its_own(self, monkeypatch):
        """The caller metered one call; the retry after a refusal is a second
        request, and at run start every concurrent call makes that same probe."""
        class _CountingLimiter:
            def __init__(self):
                self.acquired = 0

            async def acquire(self):
                self.acquired += 1

        provider = _provider(monkeypatch, _BadRequest(), _ITEM)
        provider.limiter = _CountingLimiter()
        _call(provider)
        assert len(provider._client.calls) == 2     # json_schema, then json_object
        assert provider.limiter.acquired == 1       # the first was the caller's
        assert isinstance(_make_provider("albert-api", None), _AlbertProvider)


# ── Structured output ──────────────────────────────────────────────────────

class TestStructuredOutput:

    def test_first_call_asks_for_a_closed_json_schema(self, monkeypatch):
        provider = _provider(monkeypatch, _ITEM)
        content, cost = _call(provider)

        response_format = provider._client.calls[0]["response_format"]
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == {"evaluation", "reasoning"}
        assert content == {"evaluation": 4, "reasoning": "on topic"}
        assert cost == 0.0

    def test_json_schema_call_does_not_repeat_the_schema_in_the_prompt(self, monkeypatch):
        provider = _provider(monkeypatch, _ITEM)
        _call(provider)
        sent = provider._client.calls[0]["messages"]
        assert "json schema" not in sent[-1]["content"].lower()

    def test_refusal_falls_back_to_json_object_with_the_schema_inlined(self, monkeypatch):
        provider = _provider(monkeypatch, _BadRequest("response_format not supported"), _ITEM)
        content = _call(provider)[0]

        first, second = provider._client.calls
        assert first["response_format"]["type"] == "json_schema"
        assert second["response_format"] == {"type": "json_object"}
        inlined = second["messages"][-1]["content"]
        assert json.dumps(_ReviewItem.model_json_schema()["properties"])[:20] in inlined
        assert content == {"evaluation": 4, "reasoning": "on topic"}

    def test_second_refusal_falls_back_to_plain_text(self, monkeypatch):
        provider = _provider(
            monkeypatch,
            _BadRequest("response_format not supported"),
            _BadRequest("guided decoding unavailable"),
            f"Voici le résultat :\n```json\n{_ITEM}\n```",
        )
        content = _call(provider)[0]

        assert "response_format" not in provider._client.calls[-1]
        assert content == {"evaluation": 4, "reasoning": "on topic"}

    def test_the_working_level_is_remembered(self, monkeypatch):
        provider = _provider(monkeypatch,
                             _BadRequest("response_format not supported"),
                             _ITEM, _ITEM)
        _call(provider)
        _call(provider)

        # Three calls total, not four: the second review does not re-probe
        # json_schema, it goes straight to the level that worked.
        assert len(provider._client.calls) == 3
        assert provider._client.calls[-1]["response_format"] == {"type": "json_object"}

    def test_a_real_error_is_raised_not_degraded(self, monkeypatch):
        class _Unauthorized(Exception):
            status_code = 401

        provider = _provider(monkeypatch, _Unauthorized("invalid api key"), _ITEM)
        with pytest.raises(_Unauthorized):
            _call(provider)
        assert len(provider._client.calls) == 1

    def test_schemaless_call_sends_no_response_format(self, monkeypatch):
        provider = _provider(monkeypatch, _ITEM)
        _call(provider, schema=None)
        assert "response_format" not in provider._client.calls[0]


# ── Sampling parameters ────────────────────────────────────────────────────

class TestModelArgs:

    def test_supported_params_are_forwarded(self, monkeypatch):
        provider = _provider(monkeypatch, _ITEM)
        _call(provider, model_args={"max_tokens": 200, "temperature": 0.2})
        call = provider._client.calls[0]
        assert call["max_tokens"] == 200
        assert call["temperature"] == 0.2

    def test_unsupported_params_are_dropped_with_a_warning(self, monkeypatch, capsys):
        _AlbertProvider._warned_unsupported.clear()
        provider = _provider(monkeypatch, _ITEM)
        _call(provider, model_args={"max_tokens": 200, "logit_bias": {"1": 2}})

        assert "logit_bias" not in provider._client.calls[0]
        assert "logit_bias" in capsys.readouterr().out

    def test_reasoning_effort_is_forwarded(self, monkeypatch):
        """Albert serves reasoning models, and their thinking is charged to
        max_tokens while being returned nowhere — so the one parameter that
        bounds it must reach the gateway."""
        provider = _provider(monkeypatch, _ITEM)
        _call(provider, model_args={"max_tokens": 200, "reasoning_effort": "low"})
        assert provider._client.calls[0]["reasoning_effort"] == "low"

    def test_a_deployment_refusing_reasoning_effort_drops_it_and_retries(
            self, monkeypatch, capsys):
        refusal = _BadRequest("unknown parameter: reasoning_effort")
        provider = _provider(monkeypatch, refusal, _ITEM)
        out = _call(provider, model_args={"max_tokens": 200,
                                          "reasoning_effort": "low"})

        assert out[0] == {"evaluation": 4, "reasoning": "on topic"}
        assert "reasoning_effort" in provider._client.calls[0]      # tried
        assert "reasoning_effort" not in provider._client.calls[1]  # then dropped
        assert "reasoning_effort" in capsys.readouterr().out

    def test_the_refusal_costs_the_run_nothing_else(self, monkeypatch):
        """A 400 about a sampling param must not be read as a schema refusal:
        the structured output is the thing keeping the batch parseable."""
        refusal = _BadRequest("unknown parameter: reasoning_effort")
        provider = _provider(monkeypatch, refusal, _ITEM)
        _call(provider, model_args={"reasoning_effort": "low"})
        assert provider._client.calls[1]["response_format"]["type"] == "json_schema"

    def test_a_refused_param_stays_dropped_for_later_calls(self, monkeypatch):
        refusal = _BadRequest("unknown parameter: reasoning_effort")
        provider = _provider(monkeypatch, refusal, _ITEM, _ITEM)
        _call(provider, model_args={"reasoning_effort": "low"})
        _call(provider, model_args={"reasoning_effort": "low"})
        assert "reasoning_effort" not in provider._client.calls[-1]

    def test_an_unrelated_400_is_not_silently_stripped(self, monkeypatch):
        """Stripping the request until it succeeds would hide the real error."""
        provider = _provider(monkeypatch,
                             _BadRequest("model 'albert-large' does not exist"),
                             _BadRequest("model 'albert-large' does not exist"),
                             _BadRequest("model 'albert-large' does not exist"))
        with pytest.raises(_BadRequest):
            _call(provider, model_args={"reasoning_effort": "low"})
        assert all("reasoning_effort" in c for c in provider._client.calls)
