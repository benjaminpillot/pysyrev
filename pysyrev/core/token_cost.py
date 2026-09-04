"""
A priori token counting and cost estimation for the LLM review stage.

The point of this module is to answer, *before* spending anything: how many
input and output tokens will `pysyrev <config>.yaml --stage review` send, and
what will it cost?

It does that by rebuilding the *exact* prompts the review stage will send —
:func:`pysyrev.core.llm._system_prompt`, :func:`~pysyrev.core.llm._user_prompt`
and the tool schema used for structured output are imported, not re-implemented,
so the estimate cannot drift from the code that pays the bill — and then
counting them with Anthropic's ``/v1/messages/count_tokens`` endpoint, which is
free and does not run inference.

Counting every call would mean one request per batch, so the corpus is measured
by calibration instead:

  * the per-call *fixed* cost (system prompt + batch header + tool schema) is
    counted exactly, once per reviewer;
  * the article-side cost is measured on a sample stratified by abstract length
    and allocated by character share, then extrapolated through each stratum's
    tokens-per-character ratio and the corpus' exact character counts.

Both parts are exact-per-token on the sample, so the aggregate lands within a
few percent — far better than the ubiquitous "4 characters per token" rule,
which is wrong for Claude and worse still on the long, reference-laden
"abstracts" that dominate this corpus (they run nearer 5 characters per token).

Prices are per million tokens, from platform.claude.com/docs/en/about-claude/pricing
(fetched 2026-09-01). They are a snapshot: check the page before trusting a
number to more than one significant figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

# ── Pricing table ──────────────────────────────────────────────────────────

BATCH_DISCOUNT: float = 0.5     # Batch API: −50 % on input *and* output


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, plus the cache-eligibility threshold.

    ``min_cacheable_prefix`` is the shortest prefix the model will actually
    cache: below it, a ``cache_control`` marker is silently a no-op (no error,
    no cache entry). It is *not* monotonic across generations — Haiku 4.5 needs
    4096 tokens where Opus 5 needs 512.
    """
    input:                float
    output:               float
    cache_read:           float
    cache_write_5m:       float
    cache_write_1h:       float
    min_cacheable_prefix: int


PRICING: Dict[str, ModelPrice] = {
    'claude-haiku-4-5':  ModelPrice(1.0,  5.0,  0.10, 1.25,  2.0,  4096),
    'claude-sonnet-4-6': ModelPrice(3.0, 15.0,  0.30, 3.75,  6.0,  1024),
    'claude-sonnet-5':   ModelPrice(2.0, 10.0,  0.20, 2.50,  4.0,  1024),
    'claude-opus-4-6':   ModelPrice(5.0, 25.0,  0.50, 6.25, 10.0,  4096),
    'claude-opus-4-7':   ModelPrice(5.0, 25.0,  0.50, 6.25, 10.0,  2048),
    'claude-opus-4-8':   ModelPrice(5.0, 25.0,  0.50, 6.25, 10.0,  1024),
    'claude-opus-5':     ModelPrice(5.0, 25.0,  0.50, 6.25, 10.0,   512),
    'claude-fable-5':    ModelPrice(10.0, 50.0, 1.00, 12.5, 20.0,   512),
}


def resolve_price(model_id: str) -> ModelPrice:
    """Look up a model's price, tolerating LiteLLM prefixes and date suffixes.

    ``anthropic/claude-haiku-4-5-20251001`` and ``claude-haiku-4-5`` both
    resolve to the same entry.
    """
    name = model_id.strip().lower()
    if '/' in name:                       # litellm-style "anthropic/<model>"
        name = name.rsplit('/', 1)[-1]
    if name in PRICING:
        return PRICING[name]
    # Strip a trailing -YYYYMMDD snapshot suffix.
    head = name.rsplit('-', 1)
    if len(head) == 2 and head[1].isdigit() and len(head[1]) == 8:
        if head[0] in PRICING:
            return PRICING[head[0]]
    raise KeyError(
        f"No price on file for model {model_id!r}. Known models: "
        f"{', '.join(sorted(PRICING))}. Add it to PRICING in token_cost.py."
    )


# ── Token counters ─────────────────────────────────────────────────────────

class TokenCounter:
    """Counts the input tokens of one request."""

    exact: bool = False

    def count(self, system: str, user: str,
              tools: Optional[list] = None,
              tool_choice: Optional[dict] = None) -> int:
        raise NotImplementedError

    def count_text(self, text: str) -> int:
        """Token count of a bare string (used to size model outputs)."""
        raise NotImplementedError


class AnthropicTokenCounter(TokenCounter):
    """Exact counts from ``POST /v1/messages/count_tokens`` (free, no inference)."""

    exact = True

    def __init__(self, model_id: str, client=None):
        import anthropic
        self._client = client or anthropic.Anthropic()
        self._model  = model_id

    def count(self, system, user, tools=None, tool_choice=None) -> int:
        kwargs = {}
        if system:
            kwargs['system'] = system
        if tools:
            kwargs['tools'] = tools
            if tool_choice:
                kwargs['tool_choice'] = tool_choice
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=[{'role': 'user', 'content': user or '.'}],
            **kwargs,
        )
        return int(response.input_tokens)

    def count_text(self, text: str) -> int:
        return self.count(system='', user=text)


class HeuristicTokenCounter(TokenCounter):
    """Offline fallback: characters ÷ ``chars_per_token``.

    Only used when no API key is available (``--offline``). Claude's tokenizer
    averages ~3.7 characters per token on English scientific prose; the count is
    indicative, not exact, and is flagged as such in the report.
    """

    exact = False

    def __init__(self, chars_per_token: float = 3.7, per_message_overhead: int = 12):
        self._cpt = chars_per_token
        self._overhead = per_message_overhead

    def count(self, system, user, tools=None, tool_choice=None) -> int:
        import json
        payload = (system or '') + (user or '')
        if tools:
            payload += json.dumps(tools)
        return int(len(payload) / self._cpt) + self._overhead

    def count_text(self, text: str) -> int:
        return int(len(text or '') / self._cpt)


def make_counter(model_id: str, offline: bool = False) -> TokenCounter:
    """Exact counter when the Anthropic SDK and a key are available, else heuristic."""
    if offline:
        return HeuristicTokenCounter()
    try:
        return AnthropicTokenCounter(model_id)
    except Exception:                      # missing package, missing key, …
        return HeuristicTokenCounter()


# ── Per-article token model ────────────────────────────────────────────────

@dataclass
class ArticleTokenModel:
    """Stratified estimate of the article-side token cost of a corpus.

    The corpus is cut into strata of similar character length. Within a stratum
    a few articles are counted exactly, giving that stratum's tokens-per-
    character ratio, which is then applied to the stratum's *known* total
    character count. Only the ratio is estimated — character counts are exact
    and free — and the corpus total is ``Σ ratio_j × chars_j``.

    Two alternatives were tried and are worse here. A least-squares fit of
    ``tokens ~ chars`` breaks whenever the length distribution collapses onto
    one value (truncating abstracts at a cap, say): the regressor's variance
    vanishes and the fitted slope goes somewhere arbitrary. Plain stratified
    means of tokens-per-article throw away the exact character counts, so a
    single very long abstract inside a stratum is averaged away. The ratio
    estimator has neither failure mode.
    """
    strata:      List[tuple]    # (n_articles, total_chars, tokens_per_char)
    total:       float          # tokens for the whole corpus, article side only
    n_sample:    int
    exact:       bool

    @property
    def mean_tokens(self) -> float:
        n = sum(n for n, _, _ in self.strata)
        return self.total / n if n else 0.0

    @property
    def tokens_per_char(self) -> float:
        chars = sum(c for _, c, _ in self.strata)
        return self.total / chars if chars else 0.0


def _strata(lengths: Sequence[int], n_strata: int) -> List[List[int]]:
    """Split item indices into ``n_strata`` equal-count bins of similar length."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    n_strata = max(1, min(n_strata, len(order)))
    size = len(order) / n_strata
    return [order[int(round(j * size)):int(round((j + 1) * size))]
            for j in range(n_strata)]


def fit_article_model(texts: Sequence[str], counter: TokenCounter,
                      base_prompt_fn, sample_size: int = 24,
                      n_strata: int = 8) -> ArticleTokenModel:
    """Measure the article-side token cost of ``texts`` on a stratified sample.

    ``base_prompt_fn(list_of_texts) -> (system, user, tools, tool_choice)`` must
    rebuild the request exactly as the review stage would; the fixed part is
    measured with an empty article list and subtracted out, so what is left is
    purely what the articles add.
    """
    system, user, tools, choice = base_prompt_fn([])
    fixed = counter.count(system, user, tools, choice)

    lengths = [len(t) for t in texts]
    bins = [b for b in _strata(lengths, n_strata) if b]
    bin_chars = [sum(lengths[i] for i in b) for b in bins]
    all_chars = sum(bin_chars) or 1

    strata, total, counted = [], 0.0, 0
    for b, chars in zip(bins, bin_chars):
        # Allocate the sample by *character* share, not by stratum count: the
        # estimate's variance lives where the tokens are, and in this corpus the
        # longest 5 % of abstracts carry a third of all characters.
        per_bin = min(len(b), max(2, round(sample_size * chars / all_chars)))
        # Spread the picks across the bin rather than taking its first items.
        step = len(b) / per_bin
        picks = sorted({b[min(len(b) - 1, int(i * step + step / 2))]
                        for i in range(per_bin)})
        sampled_tokens, sampled_chars = 0.0, 0
        for i in picks:
            system, user, tools, choice = base_prompt_fn([texts[i]])
            sampled_tokens += counter.count(system, user, tools, choice) - fixed
            sampled_chars  += lengths[i]
        counted += len(picks)

        if sampled_chars > 0:
            ratio = sampled_tokens / sampled_chars
            bin_tokens = ratio * chars
        else:                       # a stratum of empty texts: per-article cost only
            ratio = 0.0
            bin_tokens = len(b) * (sampled_tokens / len(picks))
        strata.append((len(b), chars, ratio))
        total += bin_tokens

    return ArticleTokenModel(strata=strata, total=total,
                             n_sample=counted, exact=counter.exact)


# ── Estimate containers ────────────────────────────────────────────────────

#: Output tokens per reviewed article, by ``reasoning`` mode. Calibrated on a
#: 50-document run of this pipeline (Haiku 4.5, ``brief`` reasoning produced a
#: ~296-character justification per article). Override with ``--calibrate``.
DEFAULT_OUTPUT_TOKENS: Dict[Optional[str], int] = {
    None:    55,
    'brief': 85,
    'cot':   230,
}


@dataclass
class GroupEstimate:
    """One (round, reviewer) pair — the unit the review stage bills in."""
    round_name:      str
    reviewer:        str
    model_id:        str
    n_items:         int
    items_per_call:  int
    n_calls:         int
    fixed_tokens:    int      # system + header + tool schema, per call
    input_tokens:    int
    output_tokens:   int
    price:           ModelPrice
    capped:          bool = False   # output estimate clipped by max_tokens

    def cost(self, batch: bool = False) -> float:
        factor = BATCH_DISCOUNT if batch else 1.0
        return factor * (self.input_tokens  * self.price.input +
                         self.output_tokens * self.price.output) / 1e6


@dataclass
class ReviewEstimate:
    """Full estimate for the review stage, plus enough detail to act on it."""
    groups:           List[GroupEstimate]
    n_docs:           int
    dataset:          Optional[str]
    escalation_rate:  float
    exact:            bool
    batch:            bool = False       # config asks for the deferred-batch transport
    article_models:   Dict[str, ArticleTokenModel] = field(default_factory=dict)
    notes:            List[str] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(g.input_tokens for g in self.groups)

    @property
    def output_tokens(self) -> int:
        return sum(g.output_tokens for g in self.groups)

    @property
    def n_calls(self) -> int:
        return sum(g.n_calls for g in self.groups)

    def cost(self, batch: bool = False) -> float:
        return sum(g.cost(batch) for g in self.groups)

    # -- reporting --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'dataset':          self.dataset,
            'n_docs':           self.n_docs,
            'escalation_rate':  self.escalation_rate,
            'token_counts':     'exact' if self.exact else 'heuristic',
            'transport':        'batch' if self.batch else 'standard',
            'input_tokens':     self.input_tokens,
            'output_tokens':    self.output_tokens,
            'n_calls':          self.n_calls,
            'cost_usd':         round(self.cost(), 4),
            'cost_usd_batch':   round(self.cost(batch=True), 4),
            'groups': [
                {
                    'round':          g.round_name,
                    'reviewer':       g.reviewer,
                    'model_id':       g.model_id,
                    'n_items':        g.n_items,
                    'n_calls':        g.n_calls,
                    'items_per_call': g.items_per_call,
                    'fixed_tokens':   g.fixed_tokens,
                    'input_tokens':   g.input_tokens,
                    'output_tokens':  g.output_tokens,
                    'cost_usd':       round(g.cost(), 4),
                }
                for g in self.groups
            ],
            'notes': self.notes,
        }

    def render(self) -> str:
        lines = []
        add = lines.append
        add('=' * 78)
        add('  LLM review — a priori token and cost estimate')
        add('=' * 78)
        add(f"  corpus          : {self.n_docs} documents")
        if self.dataset:
            add(f"  dataset         : {self.dataset}")
        add(f"  token counts    : "
            f"{'exact (count_tokens API)' if self.exact else 'HEURISTIC (offline, ±15 %)'}")
        add(f"  escalation rate : {self.escalation_rate:.0%} "
            f"(share of documents reaching rounds after the first)")
        add('')
        header = (f"  {'round':<7}{'reviewer':<13}{'model':<20}"
                  f"{'items':>7}{'calls':>7}{'in tok':>11}{'out tok':>10}{'USD':>9}")
        add(header)
        add('  ' + '-' * (len(header) - 2))
        for g in self.groups:
            add(f"  {g.round_name:<7}{g.reviewer:<13}{g.model_id:<20}"
                f"{g.n_items:>7}{g.n_calls:>7}"
                f"{_fmt_tokens(g.input_tokens):>11}{_fmt_tokens(g.output_tokens):>10}"
                f"{g.cost(self.batch):>9.2f}")
        add('  ' + '-' * (len(header) - 2))
        add(f"  {'TOTAL':<40}{sum(g.n_items for g in self.groups):>7}"
            f"{self.n_calls:>7}{_fmt_tokens(self.input_tokens):>11}"
            f"{_fmt_tokens(self.output_tokens):>10}{self.cost(self.batch):>9.2f}")
        add('')
        mark = '  ←  this run' if not self.batch else ''
        add(f"  standard API : ${self.cost():.2f}{mark}")
        add(f"  Batch API    : ${self.cost(batch=True):.2f}  "
            f"(−50 %, results within 24 h)"
            f"{'  ←  this run (review.use_batch_api: true)' if self.batch else ''}")
        add('')
        for note in self.notes:
            add(f"  • {note}")
        add('=' * 78)
        return '\n'.join(lines)


def _fmt_tokens(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(n)


# ── The estimator ──────────────────────────────────────────────────────────

def _prompt_parts(reviewer, texts: List[str], n_per_call: int):
    """Rebuild one review request exactly as :mod:`pysyrev.core.llm` sends it.

    Returns ``(system, user, tools, tool_choice)``. The tool payload mirrors
    ``_AnthropicProvider.call``: structured output goes through a forced tool,
    which adds both the schema and a per-model tool-use system prompt to every
    request — real tokens a naive estimate misses.

    ``texts`` is padded with empty articles up to ``n_per_call`` so the shape of
    the probe matches a real call: the header announces the right article count
    and every article block delimiter is present. A probe with no real text
    therefore measures the *whole* fixed cost of a call, and adding one real
    article measures exactly that article's marginal cost.

    Nothing here is rebuilt by hand — the messages come from
    :func:`~pysyrev.core.llm._prepare_call` and the tool from
    :func:`~pysyrev.core.llm.review_tool`, the same two the run itself uses.
    """
    from pysyrev.core.llm import (_prepare_call, review_tool,
                                  REVIEW_TOOL_CHOICE)

    padded = list(texts) + [''] * (n_per_call - len(texts))
    messages, _, schema = _prepare_call(padded, reviewer)
    system = next(m['content'] for m in messages if m['role'] == 'system')
    user = next(m['content'] for m in messages if m['role'] == 'user')
    return system, user, [review_tool(schema)], REVIEW_TOOL_CHOICE


def _build_reviewer_for_estimate(reviewer_config, review_config, input_description):
    """Build a runtime Reviewer for estimation only.

    ``build_reviewer`` instantiates the provider client, which needs credentials.
    The estimator only reads the reviewer's prompt-shaping fields, so a missing
    key (``--offline`` on a machine with no ``.env``) must not be fatal: fall
    back to the same dataclass with no client attached.
    """
    from pysyrev.core.llm import (build_reviewer, Reviewer, MAX_RETRIES,
                                  MAX_CONCURRENT_REQUESTS, ITEMS_PER_CALL)
    from pysyrev.review import _reviewer_kwargs

    kwargs = _reviewer_kwargs(reviewer_config, review_config, input_description)
    try:
        return build_reviewer(**kwargs)
    except Exception:
        fields = {f: kwargs.get(f) for f in Reviewer.__dataclass_fields__
                  if f != 'provider_client'}
        fields['max_retries']             = kwargs.get('max_retries')             or MAX_RETRIES
        fields['max_concurrent_requests'] = kwargs.get('max_concurrent_requests') or MAX_CONCURRENT_REQUESTS
        fields['items_per_call']          = kwargs.get('items_per_call')          or ITEMS_PER_CALL
        return Reviewer(provider_client=None, **fields)


def _synthesise_previous_rounds(dataset: pd.DataFrame, text_inputs: List[str],
                                reasoning_chars: int) -> pd.DataFrame:
    """Fill in the ``round-*_evaluation`` / ``_reasoning`` columns a later round reads.

    Rounds after the first re-send every earlier reviewer's score *and*
    justification alongside the article, so their prompts are materially longer.
    Those columns do not exist before the run, so representative values are
    substituted to keep the estimate honest.
    """
    df = dataset
    missing = [c for c in text_inputs if c not in dataset.columns]
    if not missing:
        return df
    df = dataset.copy()
    for col in missing:
        df[col] = '3' if col.endswith('_evaluation') else 'x' * reasoning_chars
    return df


def estimate_review(review_config,
                    dataset: Union[pd.DataFrame, str, None] = None,
                    *,
                    escalation_rate: float = 0.10,
                    output_tokens: Optional[Dict[str, int]] = None,
                    reasoning_chars: int = 300,
                    sample_size: int = 24,
                    offline: bool = False) -> ReviewEstimate:
    """Estimate tokens and cost for the whole review stage.

    Parameters
    ----------
    review_config : ReviewConfig
        The parsed ``review:`` section (``Config.load(...).review``).
    dataset : DataFrame or path, optional
        Corpus to review. Defaults to ``review_config.doc_dataset`` (which
        ``Config.load`` points at the latest bib run).
    escalation_rate : float
        Share of documents that fail to reach consensus in a round and are
        therefore re-reviewed in the next one. Rounds compound: round *n* sees
        ``escalation_rate ** (n-1)`` of the corpus. Measure it from a previous
        run with :func:`calibrate_from_run` rather than guessing.
    output_tokens : dict, optional
        Output tokens per article, keyed by reviewer name. Defaults to
        :data:`DEFAULT_OUTPUT_TOKENS` indexed by the reviewer's reasoning mode.
    reasoning_chars : int
        Assumed length of an earlier round's justification, re-sent to later
        rounds.
    sample_size : int
        Articles counted exactly to fit the per-article token model.
    offline : bool
        Skip the API and use the character heuristic.
    """
    from pysyrev.core.llm import build_workflow_schema, _build_texts
    from pysyrev.review import _input_description

    # -- corpus --------------------------------------------------------
    path = None
    if dataset is None:
        path = review_config.doc_dataset
        if not path:
            raise ValueError(
                "No dataset to estimate on: set review.doc_dataset in the config, "
                "run the bib stage first, or pass a dataset explicitly."
            )
        dataset = pd.read_csv(path, low_memory=False)
    elif isinstance(dataset, str):
        path = dataset
        dataset = pd.read_csv(path, low_memory=False)

    if review_config.sample_size:
        # The run's own seed when there is one, so the estimate prices the very
        # rows the review will draw; a fixed one otherwise, so re-estimating an
        # unseeded config does not move the number around.
        seed = review_config.sample_seed
        dataset = dataset.sample(review_config.sample_size,
                                 random_state=0 if seed is None else seed)

    # -- reviewers and workflow (no side effects: export is not resolved) --
    input_description = _input_description(review_config.text_inputs)
    reviewers = [
        _build_reviewer_for_estimate(rc, review_config, input_description)
        for rc in review_config.reviewers
    ]
    workflow = build_workflow_schema(review_config.workflow, reviewers,
                                     review_config.text_inputs,
                                     review_config.decision_rule)

    counters: Dict[str, TokenCounter] = {}
    article_models: Dict[str, ArticleTokenModel] = {}
    groups: List[GroupEstimate] = []
    notes: List[str] = []
    exact = not offline

    n_docs = len(dataset)
    for round_idx, round_schema in enumerate(workflow):
        share  = escalation_rate ** round_idx
        n_items = int(round(n_docs * share))
        if n_items == 0:
            continue

        # Random, not head(): later rounds see a subset of the corpus, and the
        # abstract-length distribution is heavy-tailed enough that the first N
        # rows are not representative of the N that will actually escalate.
        subset = (dataset.sample(n_items, random_state=0)
                  if n_items < n_docs else dataset)
        subset = _synthesise_previous_rounds(subset, round_schema['text_inputs'],
                                             reasoning_chars)
        texts = _build_texts(subset, round_schema['text_inputs'])

        for reviewer in round_schema['reviewers']:
            model_id = reviewer.model_id
            if model_id not in counters:
                counters[model_id] = make_counter(model_id, offline=offline)
                if not counters[model_id].exact:
                    exact = False
            counter = counters[model_id]

            n_per_call = max(1, reviewer.items_per_call)
            n_calls = math.ceil(len(texts) / n_per_call)

            base = lambda ts: _prompt_parts(reviewer, ts, n_per_call)  # noqa: E731
            key = f"{model_id}|{reviewer.name}|{round_schema['round']}"
            article_models[key] = fit_article_model(texts, counter, base,
                                                    sample_size=sample_size)
            model = article_models[key]

            system, user, tools, choice = base([])
            fixed = counter.count(system, user, tools, choice)

            input_tokens = int(round(n_calls * fixed + model.total))

            per_item = (output_tokens or {}).get(
                reviewer.name,
                DEFAULT_OUTPUT_TOKENS.get(reviewer.reasoning,
                                          DEFAULT_OUTPUT_TOKENS['brief']),
            )
            out = per_item * len(texts) + 8 * n_calls   # + JSON envelope per call
            capped = False
            if reviewer.max_tokens:
                cap = reviewer.max_tokens * n_per_call * n_calls
                if out > cap:
                    out, capped = cap, True

            groups.append(GroupEstimate(
                round_name     = str(round_schema['round']),
                reviewer       = reviewer.name,
                model_id       = model_id,
                n_items        = len(texts),
                items_per_call = n_per_call,
                n_calls        = n_calls,
                fixed_tokens   = fixed,
                input_tokens   = input_tokens,
                output_tokens  = int(out),
                price          = resolve_price(model_id),
                capped         = capped,
            ))

    use_batch = bool(getattr(review_config, 'use_batch_api', False))
    _add_notes(notes, groups, review_config, dataset, exact)
    if not use_batch:
        notes.append(
            "review.use_batch_api: true would halve this bill (same prompts, "
            "same results, collected within 24 h instead of live)."
        )
    return ReviewEstimate(groups=groups, n_docs=n_docs, dataset=path,
                          escalation_rate=escalation_rate, exact=exact,
                          batch=use_batch,
                          article_models=article_models, notes=notes)


def _add_notes(notes: List[str], groups: List[GroupEstimate],
               review_config, dataset: pd.DataFrame, exact: bool) -> None:
    """Flag the things worth acting on before the run is launched."""
    if not exact:
        notes.append("Token counts are heuristic (no API key / SDK): treat them as ±15 %.")

    missing = [c for c in review_config.text_inputs if c not in dataset.columns]
    if missing:
        notes.append(
            f"text_inputs {missing} are absent from the dataset and are silently "
            f"dropped by _build_texts — those fields are never sent to the model."
        )

    for g in groups:
        if g.fixed_tokens < g.price.min_cacheable_prefix:
            notes.append(
                f"[{g.round_name}/{g.reviewer}] the {g.fixed_tokens}-token fixed "
                f"prefix is below {g.model_id}'s {g.price.min_cacheable_prefix}-token "
                f"caching minimum: prompt caching would be a silent no-op here."
            )
            break
        overhead = 100.0 * g.n_calls * g.fixed_tokens / max(g.input_tokens, 1)
        if overhead > 15:
            notes.append(
                f"[{g.round_name}/{g.reviewer}] the per-call prefix is "
                f"{overhead:.0f} % of input tokens — raising items_per_call "
                f"amortises it further."
            )
        if g.capped:
            notes.append(
                f"[{g.round_name}/{g.reviewer}] the expected output exceeds "
                f"max_tokens; the estimate is the cap, and the run will truncate."
            )


# ── items_per_call sweep ───────────────────────────────────────────────────

def sweep_items_per_call(estimate: ReviewEstimate,
                         values: Sequence[int] = (1, 5, 10, 25, 50)) -> str:
    """What the bill would be at other ``items_per_call`` settings.

    Only the *fixed* per-call prefix changes with batching (the articles are
    sent either way), so the sweep is exact arithmetic on the measured numbers —
    no further API calls.
    """
    lines = ["  items_per_call sweep (input-side only; article tokens are invariant)",
             f"  {'per call':>9}{'calls':>8}{'in tok':>11}{'USD':>9}{'vs now':>9}"]
    now = estimate.cost()
    for v in values:
        total_in, total_calls, cost = 0, 0, 0.0
        for g in estimate.groups:
            calls = math.ceil(g.n_items / v)
            article_tokens = g.input_tokens - g.n_calls * g.fixed_tokens
            tokens = article_tokens + calls * g.fixed_tokens
            total_in += tokens
            total_calls += calls
            cost += (tokens * g.price.input + g.output_tokens * g.price.output) / 1e6
        marker = '  ←' if any(v == g.items_per_call for g in estimate.groups) else ''
        lines.append(f"  {v:>9}{total_calls:>8}{_fmt_tokens(total_in):>11}"
                     f"{cost:>9.2f}{100 * (cost - now) / now:>8.1f}%{marker}")
    return '\n'.join(lines)


# ── Calibration from a previous run ────────────────────────────────────────

def calibrate_from_run(reviewed_csv: str, model_id: str = 'claude-haiku-4-5',
                       offline: bool = False) -> dict:
    """Measure output length and escalation rate from an already-reviewed CSV.

    Returns ``{'output_tokens': {reviewer: tokens}, 'escalation_rate': float}``,
    both feedable straight into :func:`estimate_review`. This replaces the two
    weakest assumptions in the estimate with measurements from the user's own
    previous run.
    """
    df = pd.read_csv(reviewed_csv, low_memory=False)
    counter = make_counter(model_id, offline=offline)

    output_tokens: Dict[str, int] = {}
    rounds: Dict[str, float] = {}
    for col in df.columns:
        if not col.endswith('_reasoning'):
            continue
        # round-<R>_<Reviewer>_reasoning
        stem = col[:-len('_reasoning')]
        round_name, _, reviewer = stem.partition('_')
        values = df[col].dropna().astype(str)
        values = values[values.str.len() > 0]
        if values.empty:
            continue
        sample = values.sample(min(len(values), 12), random_state=0)
        mean_tokens = sum(counter.count_text(v) for v in sample) / len(sample)
        output_tokens[reviewer] = int(round(mean_tokens + 6))   # + JSON keys
        rounds[round_name] = len(values) / len(df)

    ordered = [rounds[k] for k in sorted(rounds)]
    escalation = ordered[1] / ordered[0] if len(ordered) > 1 and ordered[0] else 0.0
    return {'output_tokens': output_tokens,
            'escalation_rate': round(escalation, 4),
            'coverage_per_round': rounds}
