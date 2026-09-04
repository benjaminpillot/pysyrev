"""
One class per LLM gateway, behind a single ``call`` signature.

Supported: litellm (the default router), anthropic, openai, albert (the French
State's sovereign gateway), ollama. Each knows how to make its own request, pull
the content back out, and ask for structured output — and, where the gateway
offers one, how to reach its deferred-batch endpoint.

``build_payload`` serialises a request exactly as ``call`` would send it, so a
batched run cannot drift from a live one (nor from what
:mod:`pysyrev.core.token_cost` prices it at).
"""

import json
import os.path
from typing import List, Optional

import litellm
from pydantic import BaseModel

from pysyrev.core import batch as deferred
from pysyrev.core.llm.jsonrepair import _extract_json
from pysyrev.core.llm.ratelimit import _RateLimiter, _raise_if_truncated

litellm.drop_params = True   # silently drop params unsupported by a provider


class _ReviewItem(BaseModel):
    evaluation: int   # 1 = definitely exclude … 5 = definitely include
    reasoning:  str


class _ReviewBatch(BaseModel):
    evaluations: List[_ReviewItem]


class _BaseProvider:
    """Common interface for all LLM providers."""

    #: Shared limiter for this provider's quota, attached by `build_reviewer`
    #: (and by the labellers). Only a provider that retries a request *inside*
    #: `call` needs it: the caller metered the first attempt, so any further one
    #: is a request nobody has counted. Left None everywhere else.
    limiter: Optional['_RateLimiter'] = None

    async def call(self, messages: list, model_id: str,
                   model_args: dict, response_schema) -> tuple:
        """Make one completion call.

        `response_schema` is a Pydantic model (single-item structured output)
        or None (batch free-form JSON).  Returns (content_dict, cost_float).
        """
        raise NotImplementedError

    # -- deferred batch execution (optional capability) -------------------
    #
    # Some vendors take a pile of requests, answer them within a day, and
    # charge half. That is a property of the provider, not of the review
    # pipeline, so it is offered here and implemented by whoever has such an
    # endpoint — the orchestration around it (packing, resumption, waiting)
    # is vendor-neutral and lives in `pysyrev.core.batch`.

    def batch_backend(self):
        """The provider's deferred-batch backend, or None if it has no such endpoint."""
        return None

    def build_payload(self, messages: list, model_id: str,
                      model_args: dict, response_schema) -> dict:
        """The request body `call` would send, as the batch endpoint wants it.

        Only meaningful for a provider that returns a backend above; the point
        of building it here rather than in the batch layer is that a batched
        run must send byte-identical requests to a synchronous one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deferred batch execution."
        )


class _LiteLLMProvider(_BaseProvider):

    def __init__(self, host: Optional[str] = None,
                 timeout: Optional[float] = None):
        self._host = host
        self._timeout = timeout

    async def call(self, messages, model_id, model_args, response_schema):
        kwargs = dict(model_args)
        if response_schema is not None:
            # litellm accepts pydantic BaseModel directly for structured output
            kwargs["response_format"] = response_schema
        if self._host:
            kwargs["api_base"] = self._host
        if self._timeout is not None:
            # litellm's own default is 6000s — a hundred minutes, which is not a
            # ceiling any run wants to discover by hitting it.
            kwargs["timeout"] = self._timeout

        response = await litellm.acompletion(model=model_id, messages=messages, **kwargs)

        _raise_if_truncated(getattr(response.choices[0], "finish_reason", None),
                            model_args)
        msg = response.choices[0].message
        content = msg.content

        # Anthropic via LiteLLM routes structured output through tool_calls;
        # content can be None or "" in that case.
        if not content and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if getattr(tc, "function", None):
                    content = tc.function.arguments
                    break

        content = _extract_json(content) if isinstance(content, str) else content

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        return content, cost


def review_tool(response_schema) -> dict:
    """The forced tool that carries structured output on the Anthropic path.

    Module-level and public because it is a billed part of every request: the
    schema and the tool-use preamble it triggers are real input tokens, which
    :mod:`pysyrev.core.token_cost` must count from this definition rather than
    from a copy of it.
    """
    return {
        "name": "review",
        "description": "Structured review result",
        "input_schema": response_schema.model_json_schema(),
    }


REVIEW_TOOL_CHOICE: dict = {"type": "tool", "name": "review"}


class _AnthropicProvider(_BaseProvider):

    # Params configured by the user that this provider cannot forward. Warned
    # about once per key: silently dropping a parameter the user deliberately
    # set makes the config lie about what the run does.
    #
    # `reasoning_effort` is the one that bites. It is an OpenAI parameter; the
    # Anthropic API has no equivalent name (its `output_config.effort` is not
    # supported by every model — Haiku 4.5 rejects it). LiteLLM does accept it
    # for Anthropic models, but by translating it into a thinking budget
    # (medium -> budget_tokens 2048) billed as output tokens, which must also
    # stay under max_tokens. So the fix is to remove it, not to reroute it.
    _warned_unsupported: set = set()

    def __init__(self, host: Optional[str] = None,
                 timeout: Optional[float] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install the 'anthropic' package to use provider='anthropic'")
        kwargs = {}
        if host:
            kwargs["base_url"] = host
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def build_payload(self, messages, model_id, model_args, response_schema) -> dict:
        """Assemble the Messages API request body.

        Shared by the synchronous and the batched transports so the two cannot
        drift: a batched run must send exactly the request a live one would,
        or the estimate, the cache prefix and the results all stop agreeing.
        """
        # Anthropic's API takes system as a top-level param, not in messages.
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        # Only pass params supported by the Anthropic Messages API.
        supported = {"max_tokens", "temperature", "top_p", "top_k"}
        anthropic_args = {k: v for k, v in model_args.items()
                          if k in supported and v is not None}

        for key in model_args:
            if (key not in supported and model_args[key] is not None
                    and key not in self._warned_unsupported):
                self._warned_unsupported.add(key)
                print(f"[anthropic] '{key}' is set in the config but the "
                      f"Anthropic Messages API has no such parameter — it is "
                      f"ignored. Remove it from the reviewer.")

        payload = {"model": model_id, "system": system,
                   "messages": user_messages, **anthropic_args}

        if response_schema is not None:
            # Structured output via tool_use.
            payload["tools"] = [review_tool(response_schema)]
            payload["tool_choice"] = REVIEW_TOOL_CHOICE
        return payload

    @staticmethod
    def extract_content(response, response_schema, model_args: Optional[dict] = None):
        """Pull the review out of a Messages API response.

        Shared by both transports: a batch result carries the very same
        ``Message`` object a synchronous call returns — including its
        ``stop_reason``, so a deferred answer that ran out of output budget is
        recognised as truncated here rather than reaching the parser as a
        baffling delimiter error.
        """
        _raise_if_truncated(getattr(response, "stop_reason", None),
                            model_args or {})
        if response_schema is not None:
            content = next(
                (block.input for block in response.content if block.type == "tool_use"),
                None,
            )
            if content is None:
                raise ValueError("No tool_use block found in Anthropic response")
            return content
        return _extract_json(response.content[0].text)

    def batch_backend(self) -> '_AnthropicBatchBackend':
        return _AnthropicBatchBackend(self._client)

    async def call(self, messages, model_id, model_args, response_schema):
        payload = self.build_payload(messages, model_id, model_args, response_schema)
        response = await self._client.messages.create(**payload)
        content = self.extract_content(response, response_schema, model_args)

        try:
            cost = litellm.completion_cost(
                model=model_id,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        except Exception:
            cost = 0.0
        return content, cost


class _AnthropicBatchBackend(deferred.BatchBackend):
    """Anthropic's Message Batches endpoint, as a generic batch backend.

    Half price on input *and* output, results within 24 h. The requests are the
    ones :meth:`_AnthropicProvider.build_payload` produces and the answers are
    ordinary ``Message`` objects, so decoding reuses
    :meth:`_AnthropicProvider.extract_content` verbatim.
    """

    name = 'anthropic-batch'
    max_requests = 100_000
    max_bytes = 200 * 1024 * 1024        # documented ceiling is 256 MB

    def __init__(self, client):
        self._client = client

    async def submit(self, requests) -> str:
        created = await self._client.messages.batches.create(
            requests=[{'custom_id': r.custom_id, 'params': r.payload}
                      for r in requests]
        )
        return created.id

    async def poll(self, batch_id: str) -> deferred.BatchStatus:
        state = await self._client.messages.batches.retrieve(batch_id)
        counts = state.request_counts
        return deferred.BatchStatus(
            ended=state.processing_status == 'ended',
            done=(counts.succeeded + counts.errored
                  + counts.canceled + counts.expired),
        )

    async def fetch(self, batch_id: str, requests) -> list:
        results = []
        decoder = await self._client.messages.batches.results(batch_id)
        async for entry in decoder:
            request = requests.get(entry.custom_id)
            outcome = entry.result
            if outcome.type != 'succeeded':
                detail = getattr(outcome, 'error', None) or outcome.type
                results.append(deferred.BatchResult(entry.custom_id,
                                                 error=str(detail)))
                continue
            try:
                # A forced tool was sent iff the payload carried one, and that
                # is what says where the answer lives. The payload also carries
                # the budget the answer was written against, so a batched reply
                # cut off at max_tokens is reported as truncated — and the
                # synchronous fallback then retries it with a bigger budget,
                # which is the only recovery that works for that failure.
                payload = request.payload if request else {}
                content = _AnthropicProvider.extract_content(
                    outcome.message, payload.get('tools'), payload)
                results.append(deferred.BatchResult(entry.custom_id, content=content))
            except Exception as exc:
                results.append(deferred.BatchResult(entry.custom_id, error=str(exc)))
        return results


class _OpenAIProvider(_BaseProvider):

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None,
                 sdk_retries: Optional[int] = None,
                 timeout: Optional[float] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("Install the 'openai' package to use provider='openai'")
        kwargs = {}
        if host:
            kwargs["base_url"] = host
        if api_key:
            kwargs["api_key"] = api_key
        if sdk_retries is not None:
            kwargs["max_retries"] = sdk_retries
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = openai.AsyncOpenAI(**kwargs)

    async def call(self, messages, model_id, model_args, response_schema):
        kwargs = dict(model_args)

        if response_schema is not None and issubclass(response_schema, BaseModel):
            # Use the beta parse API: supports pydantic models and guarantees key names
            response = await self._client.beta.chat.completions.parse(
                model=model_id, messages=messages, response_format=response_schema, **kwargs,
            )
            _raise_if_truncated(getattr(response.choices[0], "finish_reason", None),
                                model_args)
            parsed = response.choices[0].message.parsed
            content = parsed.model_dump() if parsed is not None else {}
        else:
            if response_schema is not None:
                kwargs["response_format"] = response_schema
            response = await self._client.chat.completions.create(
                model=model_id, messages=messages, **kwargs,
            )
            _raise_if_truncated(getattr(response.choices[0], "finish_reason", None),
                                model_args)
            content = response.choices[0].message.content
            content = _extract_json(content) if isinstance(content, str) else content

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        return content, cost


ALBERT_DEFAULT_HOST: str = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_API_KEY_ENV:  str = "ALBERT_API_KEY"

#: Albert's published per-key quota. Unlike a paid API, this is not a soft
#: throttle to be discovered by hitting it: the gateway answers 429 outright,
#: so the run paces itself to this unless the config says otherwise.
ALBERT_REQUESTS_PER_MINUTE: int = 10

#: Seconds to wait for one Albert completion, when the config names none. The
#: OpenAI SDK's own default is 600, which is not a timeout so much as an absence
#: of one: a request queued behind a busy gateway holds a concurrency slot for
#: ten minutes before anyone hears about it. A screening call that has not
#: answered in two minutes is not going to; failing then, and re-sending, is
#: faster than waiting it out — and Albert does not bill per token, so the
#: abandoned work costs nothing but a quota slot.
#:
#: Only Albert gets a default, for that last reason. Everywhere else the tokens
#: already spent are billed, so cutting a request that would have finished is
#: money for nothing, and the number to trade that against is the user's.
ALBERT_TIMEOUT: float = 120.0


def _closed_json_schema(response_schema) -> dict:
    """JSON schema of a pydantic model, with every object closed.

    Guided decoding follows the schema literally, so an open object invites the
    model to add keys we then have to parse around; OpenAI-style ``strict``
    validators reject a schema that omits the flag outright.
    """
    schema = response_schema.model_json_schema()

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for item in node:
                close(item)

    close(schema)
    return schema


def _with_schema_instruction(messages: list, response_schema) -> list:
    """Move the schema into the prompt, for endpoints that cannot enforce it."""
    instruction = (
        "Return ONLY a JSON object valid against this JSON schema — no prose, "
        "no markdown fence:\n"
        f"{json.dumps(_closed_json_schema(response_schema))}"
    )
    sent = [dict(m) for m in messages]
    sent[-1]["content"] = f"{sent[-1]['content']}\n\n{instruction}"
    return sent


class _AlbertProvider(_OpenAIProvider):
    """Albert API — the French State's sovereign LLM gateway (Etalab / DINUM).

    The transport is OpenAI's (``/v1/chat/completions``, Bearer auth), so the
    OpenAI client is reused; the guarantees are not. Albert fronts open-weight
    models served by vLLM, where structured output is guided decoding: a given
    model may honour a full ``json_schema``, accept ``json_object`` only, or
    refuse ``response_format`` altogether — and which one is true is a property
    of the deployment, not something a config can be expected to know.

    So the call starts strict and degrades on refusal — ``json_schema`` →
    ``json_object`` (schema moved into the prompt) → plain text parsed by
    :func:`_extract_json` — and remembers the level that worked. The probe is
    paid once per run rather than on every article.
    """

    #: Structured-output levels, strictest first.
    _MODES = ("json_schema", "json_object", "text")

    #: Sampling params the gateway accepts. `reasoning_effort` is in the list
    #: and matters more here than anywhere else: Albert serves reasoning models
    #: (gpt-oss) whose thinking is charged to max_tokens but returned nowhere —
    #: the usage counts only the visible answer — so a batch can be cut off
    #: mid-JSON having "spent" a quarter of its budget on paper. Asking for less
    #: thinking is the cheap fix; raising max_tokens is the expensive one. A
    #: deployment whose model has no reasoning channel answers 400, and the
    #: parameter is then dropped for the rest of the run (see `call`).
    _SUPPORTED_ARGS = {"max_tokens", "temperature", "top_p", "seed",
                       "presence_penalty", "frequency_penalty", "stop",
                       "reasoning_effort"}
    _warned_unsupported: set = set()

    #: Params this deployment answered 400 for; dropped for the rest of the run.
    _REFUSABLE_ARGS = ("reasoning_effort",)

    def __init__(self, host: Optional[str] = None,
                 timeout: Optional[float] = None):
        # The OpenAI SDK reads OPENAI_API_KEY, which is not the key we want, so
        # Albert's own variable is read here and passed explicitly.
        api_key = os.environ.get(ALBERT_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"provider='albert' needs an API key: set {ALBERT_API_KEY_ENV} "
                f"in the .env the config points at. Keys for public-sector users "
                f"are issued at https://albert.api.etalab.gouv.fr."
            )
        # sdk_retries=0 on purpose, and only here. The OpenAI SDK retries a 429
        # twice by default, in silence and from inside the call: one logical
        # call then costs three requests against a quota the limiter believes it
        # has spent one on, and the 429 that finally surfaces has already burnt
        # two more. That is fine against an account-tier throttle, where nothing
        # is metering anything; it is not fine against Albert's ten a minute,
        # which is exactly the number this module is pacing itself to. Retrying
        # is `_review_batch`'s job — it is the layer that knows the quota, holds
        # the reviewer's other calls back and waits for the window.
        super().__init__(host=host or ALBERT_DEFAULT_HOST, api_key=api_key,
                         sdk_retries=0,
                         timeout=ALBERT_TIMEOUT if timeout is None else timeout)
        self._mode_idx = 0
        self._dropped_args: set = set()

    def _filter_args(self, model_args: dict) -> dict:
        for key, value in model_args.items():
            if (key not in self._SUPPORTED_ARGS and value is not None
                    and key not in self._warned_unsupported):
                self._warned_unsupported.add(key)
                print(f"[albert] '{key}' is set in the config but the Albert "
                      f"gateway has no such parameter — it is ignored. Remove "
                      f"it from the reviewer.")
        return {k: v for k, v in model_args.items()
                if k in self._SUPPORTED_ARGS and v is not None
                and k not in self._dropped_args}

    def _drop_refused_arg(self, exc: Exception, payload: dict) -> bool:
        """Drop a param this deployment answered 400 for, once, for the run.

        Only params in :attr:`_REFUSABLE_ARGS` and only when the error names
        them: a 400 that says nothing about the parameter is a different
        problem, and silently stripping the request until it succeeds would
        hide it.
        """
        status = (getattr(exc, "status_code", None)
                  or getattr(getattr(exc, "response", None), "status_code", None))
        if status != 400:
            return False
        text = str(exc).lower()
        for key in self._REFUSABLE_ARGS:
            if key in payload and key in text and key not in self._dropped_args:
                self._dropped_args.add(key)
                print(f"[albert] this deployment refused '{key}' ({exc}); "
                      f"dropping it for the rest of the run. Its model has no "
                      f"reasoning channel — budget max_tokens accordingly.")
                return True
        return False

    def _demote(self, exc: Exception, mode: str) -> bool:
        """Step down one structured-output level if `exc` is the endpoint
        refusing this one. Everything else — auth, rate limit, timeout, a model
        id that does not exist — is a real error and must reach the caller's
        retry logic untouched."""
        if mode == self._MODES[-1]:
            return False
        text = str(exc).lower()
        refused = (getattr(exc, "status_code", None) == 400
                   or getattr(getattr(exc, "response", None), "status_code", None) == 400
                   or any(k in text for k in ("response_format", "json_schema",
                                              "guided", "structured output",
                                              "not supported", "unsupported")))
        if not refused:
            return False
        self._mode_idx = self._MODES.index(mode) + 1
        print(f"[albert] the endpoint refused response_format={mode!r} ({exc}); "
              f"falling back to {self._MODES[self._mode_idx]!r} for the rest of "
              f"the run.")
        return True

    async def call(self, messages, model_id, model_args, response_schema):
        attempts = 0
        while True:
            # Every attempt past the first is a request the caller did not pay
            # for: it metered one call, and a demotion or a dropped param sends
            # another. At the start of a run all the concurrent calls probe
            # `json_schema` at once, so an endpoint that refuses it turns one
            # metered wave into two — which is a 429 the pacing cannot see.
            if attempts and self.limiter is not None:
                await self.limiter.acquire()
            attempts += 1

            # Rebuilt each round: a param dropped below must not come back, and
            # neither must a structured-output level already refused.
            mode = self._MODES[self._mode_idx]
            payload = self._filter_args(model_args)
            sent = messages

            if response_schema is not None:
                if mode == "json_schema":
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name":   "review",
                            "strict": True,
                            "schema": _closed_json_schema(response_schema),
                        },
                    }
                else:
                    sent = _with_schema_instruction(messages, response_schema)
                    if mode == "json_object":
                        payload["response_format"] = {"type": "json_object"}

            try:
                response = await self._client.chat.completions.create(
                    model=model_id, messages=sent, **payload,
                )
                break
            except Exception as exc:
                # Checked before _demote: that one reads any 400 as a schema
                # refusal, so a 400 about a sampling param would otherwise cost
                # the run its structured output and still not fix the request.
                if self._drop_refused_arg(exc, payload):
                    continue
                if response_schema is None or not self._demote(exc, mode):
                    raise

        _raise_if_truncated(getattr(response.choices[0], "finish_reason", None),
                            payload)
        content = response.choices[0].message.content
        content = _extract_json(content) if isinstance(content, str) else content

        # Albert does not bill per token (access is granted, not purchased), and
        # its open-weight model ids are not in LiteLLM's price map, so there is
        # no cost to report.
        return content, 0.0


def _make_provider(provider: str, host: Optional[str],
                   timeout: Optional[float] = None) -> _BaseProvider:
    """Build the runtime client for `provider`.

    `timeout` is seconds to wait for one completion, or None to leave the SDK's
    own default in place — 600 s for OpenAI and Anthropic, 6000 s for litellm.
    Those are ceilings rather than timeouts: a request queued behind a busy
    endpoint holds a concurrency slot for the whole of it. Only Albert overrides
    one by default (see :data:`ALBERT_TIMEOUT`); elsewhere the abandoned tokens
    are billed, so the trade-off belongs to the config.
    """
    if provider == "anthropic":
        return _AnthropicProvider(host=host, timeout=timeout)
    elif provider in ("albert", "albert-api"):
        return _AlbertProvider(host=host, timeout=timeout)
    elif provider in ("openai", "open-ai"):
        return _OpenAIProvider(host=host, timeout=timeout)
    elif provider == "ollama":
        return _OpenAIProvider(host=host or "http://localhost:11434/v1",
                               api_key="ollama", timeout=timeout)
    else:  # litellm — universal fallback
        return _LiteLLMProvider(host=host, timeout=timeout)


def _default_requests_per_minute(provider: str) -> Optional[int]:
    """Pacing to apply when the config names none.

    Only Albert gets one by default: its quota is a documented per-key limit
    that the gateway enforces with a 429, not a throughput hint. Everywhere
    else the limit depends on the account tier, so guessing one would silently
    slow down runs that have no need of it.
    """
    return (ALBERT_REQUESTS_PER_MINUTE
            if provider in ("albert", "albert-api") else None)
