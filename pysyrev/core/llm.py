"""
Direct-LiteLLM review functions.

Provider abstraction supports: litellm (default), anthropic, openai, albert
(the French State's sovereign gateway), ollama.
Each provider handles its own API call, content extraction, and structured
output (single-item) or free-form JSON (batch).

Key design choice: items_per_call controls how many articles are sent in a
single API request. The system prompt (backstory + criteria) is sent once
per batch rather than once per article. With items_per_call=1 the behavior
is identical to the previous per-article approach.

Requests can travel by either of two transports. The default is live: calls
go out concurrently and answers come back in seconds. The other is deferred —
the whole round is handed to the provider's batch endpoint, answered within a
day, and billed at half price. Which one is used changes nothing about what is
sent: `_prepare_call` builds the request and `build_payload` serialises it for
both, so a batched run cannot drift from a live one (nor from what
`pysyrev.core.token_cost` prices). The vendor-neutral half of that machinery —
packing, resumption, waiting — lives in `pysyrev.core.batch`.
"""

import asyncio
import json
import os.path
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import litellm
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm

from pysyrev.core import batch as deferred

litellm.drop_params = True   # silently drop params unsupported by a provider

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


# ── Response model ─────────────────────────────────────────────────────────

class _ReviewItem(BaseModel):
    evaluation: int   # 1 = definitely exclude … 5 = definitely include
    reasoning:  str


class _ReviewBatch(BaseModel):
    evaluations: List[_ReviewItem]


# ── JSON helper ────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Parse JSON from text; falls back to regex extraction if there is a preamble."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No JSON found in response: {text[:300]!r}")


# ── Provider abstraction ───────────────────────────────────────────────────

class _BaseProvider:
    """Common interface for all LLM providers."""

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

    def __init__(self, host: Optional[str] = None):
        self._host = host

    async def call(self, messages, model_id, model_args, response_schema):
        kwargs = dict(model_args)
        if response_schema is not None:
            # litellm accepts pydantic BaseModel directly for structured output
            kwargs["response_format"] = response_schema
        if self._host:
            kwargs["api_base"] = self._host

        response = await litellm.acompletion(model=model_id, messages=messages, **kwargs)

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

    def __init__(self, host: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install the 'anthropic' package to use provider='anthropic'")
        self._client = anthropic.AsyncAnthropic(**({"base_url": host} if host else {}))

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
    def extract_content(response, response_schema):
        """Pull the review out of a Messages API response.

        Also shared by both transports: a batch result carries the very same
        ``Message`` object a synchronous call returns.
        """
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
        content = self.extract_content(response, response_schema)

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
                # is what says where the answer lives.
                tools = request.payload.get('tools') if request else None
                content = _AnthropicProvider.extract_content(
                    outcome.message, tools)
                results.append(deferred.BatchResult(entry.custom_id, content=content))
            except Exception as exc:
                results.append(deferred.BatchResult(entry.custom_id, error=str(exc)))
        return results


class _OpenAIProvider(_BaseProvider):

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("Install the 'openai' package to use provider='openai'")
        kwargs = {}
        if host:
            kwargs["base_url"] = host
        if api_key:
            kwargs["api_key"] = api_key
        self._client = openai.AsyncOpenAI(**kwargs)

    async def call(self, messages, model_id, model_args, response_schema):
        kwargs = dict(model_args)

        if response_schema is not None and issubclass(response_schema, BaseModel):
            # Use the beta parse API: supports pydantic models and guarantees key names
            response = await self._client.beta.chat.completions.parse(
                model=model_id, messages=messages, response_format=response_schema, **kwargs,
            )
            parsed = response.choices[0].message.parsed
            content = parsed.model_dump() if parsed is not None else {}
        else:
            if response_schema is not None:
                kwargs["response_format"] = response_schema
            response = await self._client.chat.completions.create(
                model=model_id, messages=messages, **kwargs,
            )
            content = response.choices[0].message.content
            content = _extract_json(content) if isinstance(content, str) else content

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        return content, cost


ALBERT_DEFAULT_HOST: str = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_API_KEY_ENV:  str = "ALBERT_API_KEY"


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

    #: Sampling params the gateway accepts. `reasoning_effort` is the one that
    #: bites: OpenAI and LiteLLM take it, vLLM answers 400 for a model with no
    #: reasoning channel. Dropping it silently would make the config lie about
    #: the run, so it is warned about once per key.
    _SUPPORTED_ARGS = {"max_tokens", "temperature", "top_p", "seed",
                       "presence_penalty", "frequency_penalty", "stop"}
    _warned_unsupported: set = set()

    def __init__(self, host: Optional[str] = None):
        # The OpenAI SDK reads OPENAI_API_KEY, which is not the key we want, so
        # Albert's own variable is read here and passed explicitly.
        api_key = os.environ.get(ALBERT_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"provider='albert' needs an API key: set {ALBERT_API_KEY_ENV} "
                f"in the .env the config points at. Keys for public-sector users "
                f"are issued at https://albert.api.etalab.gouv.fr."
            )
        super().__init__(host=host or ALBERT_DEFAULT_HOST, api_key=api_key)
        self._mode_idx = 0

    def _filter_args(self, model_args: dict) -> dict:
        for key, value in model_args.items():
            if (key not in self._SUPPORTED_ARGS and value is not None
                    and key not in self._warned_unsupported):
                self._warned_unsupported.add(key)
                print(f"[albert] '{key}' is set in the config but the Albert "
                      f"gateway has no such parameter — it is ignored. Remove "
                      f"it from the reviewer.")
        return {k: v for k, v in model_args.items()
                if k in self._SUPPORTED_ARGS and v is not None}

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
        kwargs = self._filter_args(model_args)

        while True:
            mode = self._MODES[self._mode_idx]
            payload = dict(kwargs)
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
                if response_schema is None or not self._demote(exc, mode):
                    raise

        content = response.choices[0].message.content
        content = _extract_json(content) if isinstance(content, str) else content

        # Albert does not bill per token (access is granted, not purchased), and
        # its open-weight model ids are not in LiteLLM's price map, so there is
        # no cost to report.
        return content, 0.0


def _make_provider(provider: str, host: Optional[str]) -> _BaseProvider:
    if provider == "anthropic":
        return _AnthropicProvider(host=host)
    elif provider in ("albert", "albert-api"):
        return _AlbertProvider(host=host)
    elif provider in ("openai", "open-ai"):
        return _OpenAIProvider(host=host)
    elif provider == "ollama":
        return _OpenAIProvider(host=host or "http://localhost:11434/v1", api_key="ollama")
    else:  # litellm — universal fallback
        return _LiteLLMProvider(host=host)


# ── Reviewer dataclass ─────────────────────────────────────────────────────

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


def build_reviewer(name, provider, model_id,
                   host, reasoning, max_tokens,
                   temperature, reasoning_effort,
                   backstory, additional_context,
                   inclusion_criteria, exclusion_criteria,
                   input_description,
                   **kwargs) -> Reviewer:
    return Reviewer(
        name                    = name,
        model_id                = model_id,
        provider                = provider,
        provider_client         = _make_provider(provider, host),
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


# ── Core async reviewer ────────────────────────────────────────────────────

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
    the prompt builders below, keeps pricing what is actually sent.
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


async def _review_batch(
    texts: List[str], reviewer: Reviewer, semaphore: asyncio.Semaphore
) -> List[dict]:
    """Send one batch of texts in a single API call. Returns a list of evaluation dicts."""
    n_articles = len(texts)
    messages, model_args, response_schema = _prepare_call(texts, reviewer)

    # A failed multi-article call is retried once, not max_retries times. Every
    # attempt re-sends all n_articles at full price, and the dominant failure
    # here is a wrong evaluation count — the model silently dropping items —
    # which an identical retry does not fix. The split below is what actually
    # recovers, so go there instead of paying for the same batch twice.
    # Single-article calls keep the full retry budget: for them a retry is cheap
    # and the failure is usually transient.
    attempts = reviewer.max_retries if n_articles == 1 else 1

    last_exc = None
    for attempt in range(attempts):
        try:
            async with semaphore:
                raw, _ = await reviewer.provider_client.call(
                    messages, reviewer.model_id, model_args, response_schema,
                )
            return _evaluations_from_raw(raw, n_articles)
        except Exception as exc:
            last_exc = exc
            print(f"[{reviewer.name}] attempt {attempt + 1}/{attempts}: {exc}")

    # Attempts exhausted. A multi-article batch commonly fails because the model
    # returns the wrong number of evaluations at large batch sizes (it silently
    # drops items). Rather than crash the whole review, split the batch in two
    # and review each half recursively: this both realigns the counts and
    # shrinks the ask, converging on the single-item schema (which is reliable).
    # A single article that still fails is a genuine error and is raised.
    if n_articles > 1:
        mid = n_articles // 2
        print(f"[{reviewer.name}] batch of {n_articles} failed ({last_exc}); "
              f"splitting into {mid}+{n_articles - mid} and retrying")
        left = await _review_batch(texts[:mid], reviewer, semaphore)
        right = await _review_batch(texts[mid:], reviewer, semaphore)
        return left + right

    raise RuntimeError(
        f"[{reviewer.name}] failed after {attempts} attempt(s): {last_exc}"
    )


def _plan_calls(texts: List[str], reviewer: Reviewer) -> List[List[str]]:
    """Cut a reviewer's workload into calls of ``items_per_call`` articles."""
    n_per_call = max(1, reviewer.items_per_call)
    return [texts[i: i + n_per_call] for i in range(0, len(texts), n_per_call)]


async def _review_all(texts: List[str], reviewer: Reviewer) -> List[dict]:
    """Split texts into batches of items_per_call and run them concurrently."""
    semaphore = asyncio.Semaphore(reviewer.max_concurrent_requests)
    calls = _plan_calls(texts, reviewer)

    async def _indexed(idx, chunk):
        return idx, await _review_batch(chunk, reviewer, semaphore)

    ordered = [None] * len(calls)
    tasks = [_indexed(i, c) for i, c in enumerate(calls)]
    async for completed_task in tqdm(asyncio.as_completed(tasks), total=len(calls),
                                     desc=f"[{reviewer.name}] {len(texts)} articles",
                                     unit="batch"):
        idx, result = await completed_task
        ordered[idx] = result

    return [item for chunk in ordered for item in chunk]


# ── Deferred-batch reviewing ───────────────────────────────────────────────

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

        async def _resend(idx):
            reviewer, chunk = plan[idx]
            return idx, await _review_batch(chunk, reviewer, semaphores[reviewer.name])

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


# ── Workflow ───────────────────────────────────────────────────────────────

def build_workflow_schema(workflow, reviewers, text_inputs, decision_rule):
    workflow_schema = [{} for _ in range(len(workflow))]
    eval_keys_before = []   # _evaluation columns from previous rounds (used for filter)

    for round_idx, dict_from in enumerate(workflow):
        round_name     = dict_from["round"]
        reviewer_names = dict_from["reviewers"]

        eval_keys      = [f"round-{round_name}_{name}_evaluation" for name in reviewer_names]
        reasoning_keys = [f"round-{round_name}_{name}_reasoning"  for name in reviewer_names]

        workflow_schema[round_idx]["round"]       = round_name
        workflow_schema[round_idx]["reviewers"]   = [r for r in reviewers if r.name in reviewer_names]
        workflow_schema[round_idx]["text_inputs"] = list(text_inputs)

        if eval_keys_before:
            workflow_schema[round_idx]["filter"] = partial(
                eval_filter_func,
                eval_keys     = list(eval_keys_before),
                decision_rule = decision_rule,
            )

        eval_keys_before += eval_keys
        text_inputs = text_inputs + eval_keys + reasoning_keys  # new list, not in-place

    return workflow_schema


_warned_missing_inputs: set = set()


def _warn_missing_text_inputs(dataset: pd.DataFrame, text_inputs: List[str],
                              round_name: str) -> None:
    """Report text_inputs that match no column, instead of dropping them mutely.

    ``_build_texts`` skips any column it cannot find, which is what makes a
    round's later evaluation/reasoning inputs work. The same leniency hides a
    plain config typo: ask for `keywords` when the corpus column is
    `author_keywords` and the field is simply never sent to the reviewers, with
    nothing in the logs to say so.

    Warned once per round: the workflow re-runs for every checkpoint chunk, and
    a warning repeated fifty times is a warning nobody reads.
    """
    missing = [c for c in text_inputs if c not in dataset.columns]
    key = (round_name, tuple(missing))
    if missing and key not in _warned_missing_inputs:
        _warned_missing_inputs.add(key)
        print(f"[round {round_name}] text_inputs {missing} match no column in "
              f"the dataset and are not sent to the reviewers. Available: "
              f"{', '.join(sorted(dataset.columns))}")


def _build_texts(dataset: pd.DataFrame, text_inputs: List[str]) -> List[str]:
    texts = []
    for _, row in dataset.iterrows():
        parts = [f"{col}: {row[col]}"
                 for col in text_inputs
                 if col in row.index and pd.notna(row[col])]
        texts.append("\n".join(parts))
    return texts


async def _run_workflow(dataset: pd.DataFrame, workflow_schema: list,
                        batch_run: Optional[BatchRun] = None) -> pd.DataFrame:
    result = dataset.copy()
    for round_schema in workflow_schema:
        round_name  = round_schema["round"]
        reviewers   = round_schema["reviewers"]
        text_inputs = round_schema["text_inputs"]
        filter_fn   = round_schema.get("filter")

        if filter_fn:
            mask = result.apply(filter_fn, axis=1)
        else:
            mask = pd.Series(True, index=result.index)
        subset_idx = result.index[mask]

        _warn_missing_text_inputs(result, text_inputs, round_name)
        texts = _build_texts(result.loc[subset_idx], text_inputs)

        if batch_run is not None and reviewers and texts:
            # One deferred submission for the whole round, all reviewers at once.
            round_evals = await _review_round_deferred(
                reviewers, texts, round_name, batch_run)
        else:
            round_evals = {r.name: await _review_all(texts, r) for r in reviewers}

        for reviewer in reviewers:
            eval_col      = f"round-{round_name}_{reviewer.name}_evaluation"
            reasoning_col = f"round-{round_name}_{reviewer.name}_reasoning"
            evals = round_evals[reviewer.name]
            result.loc[subset_idx, eval_col]      = [e["evaluation"] for e in evals]
            result.loc[subset_idx, reasoning_col] = [e["reasoning"]  for e in evals]

    print("\n====== Finished reviewing ======\n")
    return result


# ── Score computation ──────────────────────────────────────────────────────

def compute_final_score(decision_rule, scores: pd.DataFrame):
    if decision_rule == "mean":
        return scores.mean(axis=1)
    else:  # majority
        final_scores = scores.median(axis=1)
        final_scores[final_scores == 3] = scores.iloc[final_scores == 3, -1]
        return final_scores


def eval_filter_func(row, eval_keys, decision_rule):
    score = np.asarray([int(row[eval_key])
                        for eval_key in eval_keys
                        if pd.notna(row.get(eval_key))])

    if decision_rule == "mean":
        if score.mean() == 3.:
            return True
    elif decision_rule == "majority":
        if np.count_nonzero(score > 3) == np.count_nonzero(score < 3):
            return True
        if np.count_nonzero(score == 3) / score.size >= .5:
            return True
    return False


# ── Entry points ───────────────────────────────────────────────────────────

def process_full(dataset, workflow_schema, batch_run: Optional[BatchRun] = None):
    return asyncio.run(_run_workflow(dataset, workflow_schema, batch_run))


def process_per_batch(dataset, workflow_schema, batch_size, pause, subset_file_fn):
    subsets = [dataset.iloc[i: min(i + batch_size, len(dataset)), :]
               for i in range(0, len(dataset), batch_size)]

    results = []
    for idx, subset in tqdm_sync(enumerate(subsets), total=len(subsets),
                                 desc="Batch", unit="batch"):
        if subset_file_fn is not None:
            subset_file = subset_file_fn(idx)
            if not os.path.exists(subset_file):
                reviewed_subset = asyncio.run(_run_workflow(subset, workflow_schema))
                reviewed_subset.to_csv(subset_file, index=False)
                time.sleep(pause)
            results.append(pd.read_csv(subset_file))
        else:
            reviewed_subset = asyncio.run(_run_workflow(subset, workflow_schema))
            results.append(reviewed_subset)
            time.sleep(pause)

    return pd.concat(results, ignore_index=True)


def run_review(dataset, workflow_schema, decision_rule,
               batch_size, sample_size, pause, subset_file_fn=None,
               batch_run: Optional[BatchRun] = None):

    def ds_eval_keys():
        return [f"round-{s['round']}_{r.name}_evaluation"
                for s in workflow_schema for r in s["reviewers"]]

    if sample_size:
        dataset = dataset.sample(sample_size)

    if batch_run is not None:
        # Checkpoint chunking is deliberately bypassed here. A deferred run
        # waits once per submission, so chunking a 4 600-document corpus into
        # nineteen pieces buys nineteen waits of up to 24 h for no saving —
        # and the batch is already the checkpoint, since its results stay
        # retrievable for weeks and its id is on disk.
        if batch_size and batch_size < len(dataset):
            print(f"[batch] batch_size={batch_size} is ignored in deferred-batch "
                  f"mode: each round is submitted in one go and the batch id "
                  f"itself is the restart point.")
        reviewed_dataset = process_full(dataset, workflow_schema, batch_run)
    elif batch_size and batch_size < len(dataset):
        reviewed_dataset = process_per_batch(dataset, workflow_schema, batch_size,
                                             pause, subset_file_fn)
    else:
        reviewed_dataset = process_full(dataset, workflow_schema)

    reviewed_dataset[REVIEW_SCORE] = compute_final_score(
        decision_rule, reviewed_dataset.loc[:, ds_eval_keys()]
    )
    return reviewed_dataset


# ── Topic labeling ─────────────────────────────────────────────────────────

import ast as _ast

_DEFAULT_LABELER_SYSTEM_PROMPT = (
    "You are a scientific topic labeler for a systematic literature review. "
    "Given a set of representative keywords and titles/abstracts of representative "
    "documents, generate a concise, human-readable label (5–10 words) that best "
    "describes the research topic. Return a JSON object with a single key \"label\"."
)


class _TopicLabel(BaseModel):
    label: str


def _parse_repr_list(value) -> list:
    """Parse a repr_doc column value: already a list (in-memory) or a serialised string."""
    if isinstance(value, list):
        return value
    try:
        parsed = _ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(value)]


def _build_labeler_messages(keywords: str, repr_docs: list, system_prompt: str) -> list:
    docs_text = "\n".join(
        f"{i + 1}. {doc}" for i, doc in enumerate(repr_docs) if doc
    )
    user_content = (
        f"Topic keywords: {keywords}\n\n"
        f"Representative documents:\n{docs_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


async def _label_one_topic(topic_id: int, keywords: str, repr_docs: list,
                           provider_client: _BaseProvider, model_id: str,
                           model_args: dict, system_prompt: str,
                           semaphore: asyncio.Semaphore,
                           max_retries: int) -> tuple:
    messages = _build_labeler_messages(keywords, repr_docs, system_prompt)
    last_exc = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                raw, _ = await provider_client.call(
                    messages, model_id, model_args, _TopicLabel,
                )
            label = raw.get("label", "") if isinstance(raw, dict) else str(raw)
            return topic_id, label.strip()
        except Exception as exc:
            last_exc = exc
            print(f"[topic-labeler] topic {topic_id} attempt {attempt + 1}/{max_retries}: {exc}")
    raise RuntimeError(
        f"[topic-labeler] topic {topic_id} failed after {max_retries} retries: {last_exc}"
    )


async def _label_all_topics(topic_info, config) -> dict:
    provider_client = _make_provider(config.provider, config.host)
    model_args = {}
    if config.max_tokens:
        model_args["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        model_args["temperature"] = config.temperature

    system_prompt = config.system_prompt or _DEFAULT_LABELER_SYSTEM_PROMPT
    semaphore     = asyncio.Semaphore(config.max_concurrent_requests)
    nr_docs       = config.n_repr_docs_for_labeling

    # Only use semantically useful repr_doc columns: title, abstract, author keywords.
    repr_doc_cols = [c for c in topic_info.columns if c.startswith("repr_doc_")]
    title_cols    = [c for c in repr_doc_cols if "title"   in c.lower()]
    abstract_cols = [c for c in repr_doc_cols if "abstract" in c.lower()]
    keyword_cols  = [c for c in repr_doc_cols if "keyword"  in c.lower()]
    ordered_cols  = title_cols + abstract_cols + keyword_cols

    tasks = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:        # skip outlier cluster
            continue

        keywords = str(row.get("Representation", ""))

        # Build one text snippet per representative document.
        doc_texts = []
        if ordered_cols:
            cols_values = {c: _parse_repr_list(row[c]) for c in ordered_cols if c in row}
            n = min(nr_docs, max((len(v) for v in cols_values.values()), default=0))
            for i in range(n):
                parts = []
                for col in ordered_cols:
                    vals = cols_values.get(col, [])
                    if i < len(vals) and vals[i]:
                        field_name = col.replace("repr_doc_", "").capitalize()
                        parts.append(f"{field_name}: {vals[i]}")
                if parts:
                    doc_texts.append("\n   ".join(parts))

        tasks.append(_label_one_topic(
            topic_id, keywords, doc_texts,
            provider_client, config.model_id, model_args, system_prompt,
            semaphore, config.max_retries,
        ))

    labels = {}
    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc="[topic-labeler] labeling topics", unit="topic"):
        topic_id, label = await coro
        labels[topic_id] = label

    return labels


def label_topics(topic_info, config) -> dict:
    """Generate human-readable labels for all non-outlier topics.

    Parameters
    ----------
    topic_info : pd.DataFrame
        As produced by BERTopic's get_topic_info(), enriched with repr_doc_* columns.
    config : TopicLabelerConfig

    Returns
    -------
    dict
        Mapping {topic_id (int): label (str)}.
    """
    return asyncio.run(_label_all_topics(topic_info, config))


# ── Coupling-community labeling ─────────────────────────────────────────────

_DEFAULT_CLUSTER_LABELER_SYSTEM_PROMPT = (
    "You are a scientific topic labeler for a systematic literature review. "
    "You are given one community of a bibliographic-coupling network: papers that "
    "cite the same references and therefore share an intellectual base. The "
    "community is described by its distinguishing TF-IDF terms and, when "
    "available, a few of its most central paper titles. Generate a concise, "
    "human-readable label (5–10 words) that names the sub-theme these papers form. "
    "Return a JSON object with a single key \"label\"."
)


def _build_cluster_labeler_messages(terms: list, titles: list,
                                     system_prompt: str) -> list:
    parts = [f"Distinguishing TF-IDF terms: {', '.join(terms)}"]
    if titles:
        docs_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles) if t)
        parts.append(f"Representative paper titles:\n{docs_text}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]


async def _label_one_cluster(comm_id: int, terms: list, titles: list,
                             provider_client: _BaseProvider, model_id: str,
                             model_args: dict, system_prompt: str,
                             semaphore: asyncio.Semaphore,
                             max_retries: int) -> tuple:
    messages = _build_cluster_labeler_messages(terms, titles, system_prompt)
    last_exc = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                raw, _ = await provider_client.call(
                    messages, model_id, model_args, _TopicLabel,
                )
            label = raw.get("label", "") if isinstance(raw, dict) else str(raw)
            return comm_id, label.strip()
        except Exception as exc:
            last_exc = exc
            print(f"[cluster-labeler] community {comm_id} "
                  f"attempt {attempt + 1}/{max_retries}: {exc}")
    raise RuntimeError(
        f"[cluster-labeler] community {comm_id} failed after "
        f"{max_retries} retries: {last_exc}"
    )


async def _label_all_clusters(terms: dict, config, repr_docs=None) -> dict:
    provider_client = _make_provider(config.provider, config.host)
    model_args = {}
    if config.max_tokens:
        model_args["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        model_args["temperature"] = config.temperature

    system_prompt = config.system_prompt or _DEFAULT_CLUSTER_LABELER_SYSTEM_PROMPT
    semaphore     = asyncio.Semaphore(config.max_concurrent_requests)
    nr_docs       = config.n_repr_docs_for_labeling
    repr_docs     = repr_docs or {}

    tasks = []
    for comm_id in sorted(terms):
        titles = [t for t in (repr_docs.get(comm_id) or [])[:nr_docs] if t]
        tasks.append(_label_one_cluster(
            int(comm_id), list(terms[comm_id]), titles,
            provider_client, config.model_id, model_args, system_prompt,
            semaphore, config.max_retries,
        ))

    labels = {}
    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc="[cluster-labeler] labeling communities",
                           unit="community"):
        comm_id, label = await coro
        labels[comm_id] = label

    return labels


def label_clusters(terms: dict, config, repr_docs=None) -> dict:
    """Generate human-readable labels for bibliographic-coupling communities.

    Parameters
    ----------
    terms : dict
        Mapping ``{community_id (int): [distinguishing TF-IDF terms]}`` — as
        produced by the coupling network's per-community sub-theme extraction.
    config : TopicLabelerConfig
        The same labeler config used for topics.
    repr_docs : dict, optional
        Mapping ``{community_id (int): [representative paper titles]}`` to anchor
        the label. Only the first ``n_repr_docs_for_labeling`` are used.

    Returns
    -------
    dict
        Mapping ``{community_id (int): label (str)}``.
    """
    return asyncio.run(_label_all_clusters(terms, config, repr_docs))
