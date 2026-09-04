"""
Human-readable labels for topics and for network communities.

Both are the same errand — a handful of short calls asking a model to name a
cluster from its keywords and a few representative documents — so both go
through one driver, :func:`_label_all`. They are few calls, which is why they
used to be unpaced; but they are drawn on the key the review has just been
pacing itself against, so they take their turn in the same shared quota.

Results are cached by :mod:`pysyrev.core.topic_labels`, so the labeller runs
once per partition rather than once per report.
"""

import asyncio
import ast as _ast
from typing import Optional

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from pysyrev.core.llm.providers import (_BaseProvider,
                                        _default_requests_per_minute,
                                        _make_provider)
from pysyrev.core.llm.ratelimit import (RATE_LIMIT_MAX_WAITS, _RateLimiter,
                                        _is_rate_limited, _limiter_for_quota,
                                        _rate_limit_delay)


def _labeler_limiter(config) -> Optional['_RateLimiter']:
    """The limiter for a labelling stage, from the same registry as the review.

    A labelling run is a handful of short calls, which is exactly why it had
    none: too few to be worth pacing on their own. But they are drawn on the
    key the review has just been pacing itself against, so on their own they
    are not the question — the quota is.
    """
    return _limiter_for_quota(
        config.provider, config.host,
        getattr(config, "requests_per_minute", None)
        or _default_requests_per_minute(config.provider))


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


async def _labeler_call(tag: str, what: str, messages: list,
                        provider_client: _BaseProvider, model_id: str,
                        model_args: dict, semaphore: asyncio.Semaphore,
                        limiter: Optional[_RateLimiter],
                        max_retries: int) -> str:
    """One labelling call, with the review path's rate-limit recovery.

    Shared by both labellers because a 429 needs the same answer wherever it is
    met: an immediate retry against a gateway that is refusing requests is not a
    recovery, it is a second refusal, and it used to burn both attempts and lose
    the label. So a rate limit waits — holding back the other labelling calls
    through the shared limiter — instead of consuming an attempt.
    """
    last_exc = None
    attempt = waits = 0
    while attempt < max_retries:
        try:
            # Quota taken inside the semaphore, immediately before the call, so
            # the window records a send rather than an intention to send.
            async with semaphore:
                if limiter is not None:
                    await limiter.acquire()
                raw, _ = await provider_client.call(
                    messages, model_id, model_args, _TopicLabel,
                )
            label = raw.get("label", "") if isinstance(raw, dict) else str(raw)
            return label.strip()
        except Exception as exc:
            last_exc = exc

            if _is_rate_limited(exc) and waits < RATE_LIMIT_MAX_WAITS:
                delay = _rate_limit_delay(exc, waits)
                waits += 1
                if limiter is not None:
                    await limiter.penalise(delay)
                print(f"[{tag}] {what}: rate-limited ({exc}); waiting "
                      f"{delay:.0f}s ({waits}/{RATE_LIMIT_MAX_WAITS})")
                await asyncio.sleep(delay)
                continue

            attempt += 1
            print(f"[{tag}] {what} attempt {attempt}/{max_retries}: {exc}")

    raise RuntimeError(
        f"[{tag}] {what} failed after {max_retries} retries: {last_exc}"
    )



async def _label_all(items, config, *, tag: str, default_system_prompt: str,
                     build_messages, desc: str, unit: str) -> dict:
    """Shared driver for both labellers — returns ``{item_id: label}``.

    *items* is a list of ``(item_id, payload)``; ``build_messages(payload,
    system_prompt)`` turns one payload into the chat messages. Everything around
    that is the same whichever thing is being labelled: the provider, the
    concurrency cap, the shared rate limiter the reviewers were pacing
    themselves against, the 429 recovery in :func:`_labeler_call`, the bar.
    """
    provider_client = _make_provider(config.provider, config.host,
                                     getattr(config, "timeout", None))
    model_args = {}
    if config.max_tokens:
        model_args["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        model_args["temperature"] = config.temperature

    system_prompt = config.system_prompt or default_system_prompt
    semaphore     = asyncio.Semaphore(config.max_concurrent_requests)
    # Same registry as the review: labelling draws on the key the
    # reviewers have just been pacing themselves against.
    limiter       = _labeler_limiter(config)
    provider_client.limiter = limiter

    async def label_one(item_id, payload):
        label = await _labeler_call(
            tag, f"{unit} {item_id}", build_messages(payload, system_prompt),
            provider_client, config.model_id, model_args, semaphore, limiter,
            config.max_retries,
        )
        return item_id, label

    tasks = [label_one(item_id, payload) for item_id, payload in items]
    labels = {}
    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc=f"[{tag}] {desc}", unit=unit):
        item_id, label = await coro
        labels[item_id] = label

    return labels


def _topic_payloads(topic_info, nr_docs: int) -> list:
    """``(topic_id, (keywords, doc_texts))`` for every non-outlier topic."""
    # Only use semantically useful repr_doc columns: title, abstract, author keywords.
    repr_doc_cols = [c for c in topic_info.columns if c.startswith("repr_doc_")]
    title_cols    = [c for c in repr_doc_cols if "title"    in c.lower()]
    abstract_cols = [c for c in repr_doc_cols if "abstract" in c.lower()]
    keyword_cols  = [c for c in repr_doc_cols if "keyword"  in c.lower()]
    ordered_cols  = title_cols + abstract_cols + keyword_cols

    items = []
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

        items.append((topic_id, (keywords, doc_texts)))
    return items


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
    return asyncio.run(_label_all(
        _topic_payloads(topic_info, config.n_repr_docs_for_labeling), config,
        tag                   = "topic-labeler",
        default_system_prompt = _DEFAULT_LABELER_SYSTEM_PROMPT,
        build_messages        = lambda p, sp: _build_labeler_messages(*p, sp),
        desc                  = "labeling topics",
        unit                  = "topic",
    ))


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
    nr_docs   = config.n_repr_docs_for_labeling
    repr_docs = repr_docs or {}
    items = [
        (int(comm_id), (list(terms[comm_id]),
                        [t for t in (repr_docs.get(comm_id) or [])[:nr_docs] if t]))
        for comm_id in sorted(terms)
    ]
    return asyncio.run(_label_all(
        items, config,
        tag                   = "cluster-labeler",
        default_system_prompt = _DEFAULT_CLUSTER_LABELER_SYSTEM_PROMPT,
        build_messages        = lambda p, sp: _build_cluster_labeler_messages(*p, sp),
        desc                  = "labeling communities",
        unit                  = "community",
    ))
