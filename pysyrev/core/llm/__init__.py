"""
LLM-assisted screening and labelling.

Provider abstraction supports: litellm (default), anthropic, openai, albert
(the French State's sovereign gateway), ollama. Each provider handles its own
API call, content extraction, and structured output (single-item) or free-form
JSON (batch).

Key design choice: ``items_per_call`` controls how many articles are sent in a
single API request. The system prompt (backstory + criteria) is sent once per
batch rather than once per article. With ``items_per_call=1`` the behaviour is
identical to the previous per-article approach.

Requests can travel by either of two transports. The default is live: calls go
out concurrently and answers come back in seconds. The other is deferred — the
whole round is handed to the provider's batch endpoint, answered within a day,
and billed at half price. Which one is used changes nothing about what is sent:
``_prepare_call`` builds the request and ``build_payload`` serialises it for
both, so a batched run cannot drift from a live one (nor from what
:mod:`pysyrev.core.token_cost` prices).

Layout
------
:mod:`~pysyrev.core.llm.ratelimit`
    Failure kinds (truncation, 429, timeout) and the shared per-quota limiter.
:mod:`~pysyrev.core.llm.jsonrepair`
    Pulling JSON back out of a model's answer.
:mod:`~pysyrev.core.llm.providers`
    One class per gateway, behind a single ``call`` signature.
:mod:`~pysyrev.core.llm.reviewer`
    The reviewer persona, the prompt, and how one call is made and retried.
:mod:`~pysyrev.core.llm.batch_run`
    The deferred transport.
:mod:`~pysyrev.core.llm.workflow`
    Rounds, votes, and the entry points the review stage calls.
:mod:`~pysyrev.core.llm.labeling`
    Topic and community labels.

Everything public is re-exported here, so ``pysyrev.core.llm`` remains the
single import path it has always been.
"""

from pysyrev.core.llm.ratelimit import (  # noqa: F401
    RATE_LIMIT_BASE_DELAY, RATE_LIMIT_MAX_DELAY, RATE_LIMIT_MAX_WAITS,
    RATE_LIMIT_WINDOW, TIMEOUT_MAX_RETRIES, TRUNCATION_MAX_BUMPS, _LIMITERS,
    TruncatedResponse, _is_rate_limited, _is_timeout, _limiter_for_quota,
    _raise_if_truncated, _RateLimiter, _rate_limit_delay)
from pysyrev.core.llm.jsonrepair import _extract_json, _unclosed_json  # noqa: F401
from pysyrev.core.llm.providers import (  # noqa: F401
    ALBERT_API_KEY_ENV, ALBERT_DEFAULT_HOST, ALBERT_REQUESTS_PER_MINUTE,
    ALBERT_TIMEOUT, REVIEW_TOOL_CHOICE, _AlbertProvider, _AnthropicBatchBackend,
    _AnthropicProvider, _BaseProvider, _LiteLLMProvider, _OpenAIProvider,
    _ReviewBatch, _ReviewItem, _closed_json_schema,
    _default_requests_per_minute, _make_provider, _with_schema_instruction,
    review_tool)
from pysyrev.core.llm.reviewer import (  # noqa: F401
    ITEMS_PER_CALL, MAX_CONCURRENT_REQUESTS, MAX_RETRIES, REVIEW_SCORE,
    Reviewer, _build_model_args, _bumped_max_tokens, _evaluations_from_raw,
    _normalize_evaluations, _parse_evaluation, _plan_calls, _prepare_call,
    _rate_limiter_for, _review_all, _review_batch, _system_prompt, _user_prompt,
    build_reviewer)
from pysyrev.core.llm.batch_run import (  # noqa: F401
    BatchRun, _review_round_deferred, check_batch_support)
from pysyrev.core.llm.workflow import (  # noqa: F401
    _build_texts, _run_workflow, build_workflow_schema, compute_final_score,
    eval_filter_func, process_full, process_per_batch, run_review)
from pysyrev.core.llm.labeling import (  # noqa: F401
    _labeler_limiter, label_clusters, label_topics)
