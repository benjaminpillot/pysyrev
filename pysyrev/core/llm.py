"""
Direct-LiteLLM review functions.

Provider abstraction supports: litellm (default), anthropic, openai, ollama.
Each provider handles its own API call, content extraction, and structured
output (single-item) or free-form JSON (batch).

Key design choice: items_per_call controls how many articles are sent in a
single API request. The system prompt (backstory + criteria) is sent once
per batch rather than once per article. With items_per_call=1 the behavior
is identical to the previous per-article approach.
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
from tqdm.asyncio import tqdm

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


class _AnthropicProvider(_BaseProvider):

    def __init__(self, host: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install the 'anthropic' package to use provider='anthropic'")
        self._client = anthropic.AsyncAnthropic(**({"base_url": host} if host else {}))

    async def call(self, messages, model_id, model_args, response_schema):
        # Anthropic's API takes system as a top-level param, not in messages.
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        # Only pass params supported by the Anthropic Messages API.
        supported = {"max_tokens", "temperature", "top_p", "top_k"}
        anthropic_args = {k: v for k, v in model_args.items()
                          if k in supported and v is not None}

        if response_schema is not None:
            # Structured output via tool_use.
            tool_def = {
                "name": "review",
                "description": "Structured review result",
                "input_schema": response_schema.model_json_schema(),
            }
            response = await self._client.messages.create(
                model=model_id, system=system, messages=user_messages,
                tools=[tool_def], tool_choice={"type": "tool", "name": "review"},
                **anthropic_args,
            )
            content = next(
                (block.input for block in response.content if block.type == "tool_use"),
                None,
            )
            if content is None:
                raise ValueError("No tool_use block found in Anthropic response")
        else:
            response = await self._client.messages.create(
                model=model_id, system=system, messages=user_messages, **anthropic_args,
            )
            content = _extract_json(response.content[0].text)

        try:
            cost = litellm.completion_cost(
                model=model_id,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        except Exception:
            cost = 0.0
        return content, cost


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


def _make_provider(provider: str, host: Optional[str]) -> _BaseProvider:
    if provider == "anthropic":
        return _AnthropicProvider(host=host)
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

async def _review_batch(
    texts: List[str], reviewer: Reviewer, semaphore: asyncio.Semaphore
) -> List[dict]:
    """Send one batch of texts in a single API call. Returns a list of evaluation dicts."""
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

    last_exc = None
    for attempt in range(reviewer.max_retries):
        try:
            async with semaphore:
                raw, _ = await reviewer.provider_client.call(
                    messages, reviewer.model_id, model_args, response_schema,
                )
            if n_articles == 1:
                return [{"evaluation": int(raw["evaluation"]), "reasoning": str(raw["reasoning"])}]

            # Accept {"evaluations": [...]} or a bare list
            if isinstance(raw, list):
                evals = raw
            elif isinstance(raw, dict):
                evals = raw.get("evaluations") or []
            else:
                raise ValueError(f"Unexpected response type: {type(raw)}")

            if len(evals) != n_articles:
                raise ValueError(f"Expected {n_articles} evaluations, got {len(evals)}")
            return [{"evaluation": int(e["evaluation"]), "reasoning": str(e["reasoning"])}
                    for e in evals]
        except Exception as exc:
            last_exc = exc
            print(f"[{reviewer.name}] attempt {attempt + 1}/{reviewer.max_retries}: {exc}")
    raise RuntimeError(
        f"[{reviewer.name}] failed after {reviewer.max_retries} retries: {last_exc}"
    )


async def _review_all(texts: List[str], reviewer: Reviewer) -> List[dict]:
    """Split texts into batches of items_per_call and run them concurrently."""
    semaphore = asyncio.Semaphore(reviewer.max_concurrent_requests)
    n_articles_per_call = reviewer.items_per_call
    batches = [texts[i: i + n_articles_per_call]
               for i in range(0, len(texts), n_articles_per_call)]

    async def _indexed(idx, batch):
        return idx, await _review_batch(batch, reviewer, semaphore)

    ordered = [None] * len(batches)
    tasks = [_indexed(i, b) for i, b in enumerate(batches)]
    async for completed_task in tqdm(asyncio.as_completed(tasks), total=len(batches),
                                     desc=f"[{reviewer.name}] {len(texts)} articles",
                                     unit="batch"):
        idx, result = await completed_task
        ordered[idx] = result

    return [item for batch in ordered for item in batch]


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


def _build_texts(dataset: pd.DataFrame, text_inputs: List[str]) -> List[str]:
    texts = []
    for _, row in dataset.iterrows():
        parts = [f"{col}: {row[col]}"
                 for col in text_inputs
                 if col in row.index and pd.notna(row[col])]
        texts.append("\n".join(parts))
    return texts


async def _run_workflow(dataset: pd.DataFrame, workflow_schema: list) -> pd.DataFrame:
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

        texts = _build_texts(result.loc[subset_idx], text_inputs)

        for reviewer in reviewers:
            eval_col      = f"round-{round_name}_{reviewer.name}_evaluation"
            reasoning_col = f"round-{round_name}_{reviewer.name}_reasoning"
            evals = await _review_all(texts, reviewer)
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

def process_full(dataset, workflow_schema):
    return asyncio.run(_run_workflow(dataset, workflow_schema))


def process_per_batch(dataset, workflow_schema, batch_size, pause, subset_file_fn):
    subsets = [dataset.iloc[i: min(i + batch_size, len(dataset)), :]
               for i in range(0, len(dataset), batch_size)]

    results = []
    for idx, subset in enumerate(subsets):
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
               batch_size, sample_size, pause, subset_file_fn=None):

    def ds_eval_keys():
        return [f"round-{s['round']}_{r.name}_evaluation"
                for s in workflow_schema for r in s["reviewers"]]

    if sample_size:
        dataset = dataset.sample(sample_size)

    if batch_size and batch_size < len(dataset):
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
