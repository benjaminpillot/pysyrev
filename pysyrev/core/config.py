"""
Configuration loader.

Reads a YAML file and produces typed configuration dataclasses. These
classes know nothing about the business model — they only mirror the YAML
structure. The bridge from config to the runtime model lives in
topic_model.py via TopicModel.from_config().

Naming convention: every config dataclass ends with `Config` so it is
unambiguous when imported alongside the runtime classes (e.g. UMAPConfig
vs UmapModel).

Environment variables: any string in the YAML may contain ${VAR} references.
They are resolved at load time using the environment, augmented with the
contents of the .env file pointed to by the root-level `env:` key (if any).
"""
import dataclasses
import datetime
import os
import re
from dataclasses import dataclass, fields, field
from typing import List, Optional, Union

import yaml
from dotenv import load_dotenv


# Pattern to resolve ${ENV_VAR} references inside YAML string values.
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _make_run_dir(export_dir: str, run_name: Union[str, None]) -> tuple:
    """Return ``(run_name, run_dir)``, creating the directory.

    If *run_name* is blank a timestamp (``YYYY-MM-DDTHHMMSS``) is generated.
    ``exist_ok=True`` so that re-opening an existing run is allowed.
    """
    if not run_name:
        run_name = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    run_dir = os.path.join(export_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def _find_latest_dir(export_dir: str) -> Union[str, None]:
    """Return the most recently modified subdirectory of *export_dir*."""
    if not os.path.isdir(export_dir):
        return None
    candidates = [
        (entry.stat().st_mtime, entry.path)
        for entry in os.scandir(export_dir)
        if entry.is_dir()
    ]
    return max(candidates, default=(None, None))[1]


def _find_latest_file(export_dir: str, filename: str) -> Union[str, None]:
    """Return *filename* in the most recently modified subdirectory of *export_dir*.

    Handles two layouts:
    - flat:              export_dir/filename
    - timestamped dirs:  export_dir/<run_name>/filename
    """
    if not os.path.isdir(export_dir):
        return None
    flat = os.path.join(export_dir, filename)
    if os.path.isfile(flat):
        return flat
    latest_dir = _find_latest_dir(export_dir)
    if latest_dir is None:
        return None
    candidate = os.path.join(latest_dir, filename)
    return candidate if os.path.isfile(candidate) else None



def _resolve_env_vars(node) -> dict:
    """Recursively walk a YAML structure (dict / list / scalar) and replace
    ${VAR} occurrences in string values with os.environ[VAR]. Raises a
    ValueError if a referenced variable is missing."""
    if isinstance(node, dict):
        return {k: _resolve_env_vars(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_env_vars(v) for v in node]
    if isinstance(node, str):
        def replace(match):
            var_name = match.group(1)
            if var_name not in os.environ:
                raise ValueError(
                    f"Environment variable {var_name!r} is referenced in the "
                    f"config but is not set (check your .env file)"
                )
            return os.environ[var_name]
        return _ENV_VAR_PATTERN.sub(replace, node)
    return node


@dataclass
class ConfigField:
    """Base for config dataclasses. If a YAML entry sets a field to None
    explicitly, fall back to the dataclass default (when one is defined)."""

    def __post_init__(self):
        for field_ in fields(self):
            if (not isinstance(field_.default, dataclasses._MISSING_TYPE)
                    and getattr(self, field_.name) is None):
                setattr(self, field_.name, field_.default)


@dataclass
class WosApiConfig(ConfigField):
    """Configuration for retrieving WoS records via the Expanded API."""
    api_key:   str   # typically `${WOS_API_KEY}` -> resolved from .env
    query:     str   # WoS Query Language, e.g. 'TS=("agent-based") AND PY=2015-2024'
    cache_dir: Union[None, str] = None  # local cache; None = no caching


@dataclass
class WosSourceConfig(ConfigField):
    """One WoS source: either a file path, or an API config. Exactly one
    of `file` / `api` must be set."""
    source: str = 'file'  # 'file' or 'api'
    file:   Union[None, str] = None
    api:    Union[None, WosApiConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.source == 'api' and isinstance(self.api, dict):
            self.api = WosApiConfig(**self.api)
        if self.source == 'file' and not self.file:
            raise ValueError("WoS source is 'file' but no `file:` path is set")
        if self.source == 'api' and self.api is None:
            raise ValueError("WoS source is 'api' but no `api:` block is set")
        if self.source not in ('file', 'api'):
            raise ValueError(f"Unknown WoS source {self.source!r}; expected 'file' or 'api'")


@dataclass
class AbstractCompletionConfig(ConfigField):
    """One abstract-completion step: recover missing abstracts from an external
    provider, looked up by DOI.

    Declared under ``bib.abstract_completion`` as a list — completion is
    corpus-level, not tied to any one source, since any source can leave
    abstracts missing. Each entry names a ``provider`` and its credentials.
    Entries run in order on the merged corpus (before no-abstract rows are
    dropped), so a later provider only fills abstracts still missing after the
    earlier ones. Only the ``abstract`` column is touched — references/IDs used
    for coupling networks are left intact.

    ``provider`` selects the backend: ``wos`` is the only one implemented today;
    ``scopus`` / ``pubmed`` would register their own backend the same way.
    """
    provider:  Union[None, str] = None   # 'wos' today; 'scopus' / 'pubmed' planned
    api_key:   Union[None, str] = None    # typically `${WOS_API_KEY}` etc.
    cache_dir: Union[None, str] = None    # optional provider session cache

    def __post_init__(self):
        super().__post_init__()
        if not self.provider:
            raise ValueError(
                "an abstract_completion entry needs a `provider:` (e.g. 'wos')")
        if not self.api_key:
            raise ValueError(
                f"abstract_completion provider {self.provider!r} needs an `api_key:`")


@dataclass
class SeedExpansionConfig(ConfigField):
    """Provider-agnostic settings of a ``source: seed`` block.

    Seed expansion builds the corpus from a handful of *seed papers* instead of
    a single query: the pool is the union of the seeds, the works citing them,
    the works they cite, and optional text queries (see
    :mod:`pysyrev.core.seed_expansion`). Use it when a subfield has no reliable
    keyword handle.

    ``file``
        Text file listing the seeds, **one DOI per line** (an OpenAlex ``Wxxxx``
        id also works). Blank lines and ``#`` comments are ignored.
    ``seeds``
        Inline seed list, merged with the ones read from ``file``.
    ``queries``
        Title/abstract searches in the field's own vocabulary (10-25 is a good
        range) — the recall backstop for works with no citation link to a seed.
        Each entry is one query in the provider's search syntax, run
        separately; the results are unioned. Leave blank to expand on citations
        only.
    ``year_min`` / ``year_max``
        Publication-year window used as a *retrieval* bound on the arms whose
        provider supports it. Definitive filtering stays with ``bib.extract``,
        so seeds outside the window still drive the expansion.
    ``use_forward`` / ``use_backward``
        Toggle the citation arms. Both off leaves a query-only pool.
    ``max_per_seed`` / ``max_per_query``
        Caps on the records pulled per seed (forward citations) and per query.
        Raise for exhaustive recall, lower for a quick draft.

    The pool is a set of *candidates*, deliberately over-collected: screening it
    down to the corpus is the ``review`` stage's job.
    """
    api_key:       str
    file:          Union[None, str]       = None
    seeds:         Union[None, List[str]] = None
    queries:       Union[None, List[str]] = None
    year_min:      Union[None, int]       = None
    year_max:      Union[None, int]       = None
    use_forward:   bool                   = True
    use_backward:  bool                   = True
    max_per_seed:  int                    = 600
    max_per_query: int                    = 400
    cache_dir:     Union[None, str]       = None

    #: Former key -> current key. ``phrases`` was renamed because the entries
    #: are not phrase-matched: their terms are AND-ed, the way any query to the
    #: provider would be.
    RENAMED_KEYS = {'phrases': 'queries', 'max_per_phrase': 'max_per_query'}

    @classmethod
    def from_dict(cls, block: dict) -> 'SeedExpansionConfig':
        """Build from a YAML block, accepting the pre-rename key spellings."""
        renamed = {cls.RENAMED_KEYS.get(k, k): v for k, v in block.items()}
        return cls(**renamed)

    def __post_init__(self):
        super().__post_init__()
        if not self.file and not self.seeds:
            raise ValueError(
                "A seed source needs a `file:` (one DOI per line) or an inline "
                "`seeds:` list.")

    def seed_tokens(self) -> List[str]:
        """The declared seeds: those read from ``file``, then the inline
        ``seeds``, deduplicated and in declaration order."""
        from pysyrev.core.seed_expansion import read_seed_file
        tokens = read_seed_file(self.file) if self.file else []
        tokens += [str(s).strip() for s in (self.seeds or []) if str(s).strip()]
        return list(dict.fromkeys(tokens))


@dataclass
class OpenAlexSeedConfig(SeedExpansionConfig):
    """OpenAlex seed expansion — the generic settings plus the polite-pool
    email (see :class:`SeedExpansionConfig`)."""
    email: Union[None, str] = None       # optional, for the polite pool


@dataclass
class OpenAlexApiConfig(ConfigField):
    """Configuration for retrieving works via the OpenAlex API.

    Either `query` (free-text BM25) or `filters` (structured) must be set
    (both can also be combined). `email` enables the polite pool — it is
    optional but strongly recommended for non-trivial usage.
    """
    api_key:   str
    email:     Union[None, str] = None       # optional, for the polite pool
    query:     Union[None, str] = None       # full-text search (BM25 on title+abstract)
    filters:   Union[None, dict] = None      # structured filters; e.g. {'publication_year': '2015-2024'}
    cache_dir: Union[None, str] = None       # local cache; None = no caching


@dataclass
class OpenAlexSourceConfig(ConfigField):
    """One OpenAlex source: a file path, an API query, or a seed expansion.
    Exactly one of `file` / `api` / `seed` must be set.

    Abstract completion is no longer declared here: it is corpus-level, under
    ``bib.abstract_completion`` (see :class:`AbstractCompletionConfig`)."""
    source:         str = 'file'  # 'file', 'api' or 'seed'
    file:           Union[None, str] = None
    api:            Union[None, OpenAlexApiConfig] = None
    seed:           Union[None, OpenAlexSeedConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.source == 'api' and isinstance(self.api, dict):
            self.api = OpenAlexApiConfig(**self.api)
        if self.source == 'seed' and isinstance(self.seed, dict):
            self.seed = OpenAlexSeedConfig.from_dict(self.seed)
        if self.source == 'file' and not self.file:
            raise ValueError("OpenAlex source is 'file' but no `file:` path is set")
        if self.source == 'api' and self.api is None:
            raise ValueError("OpenAlex source is 'api' but no `api:` block is set")
        if self.source == 'seed' and self.seed is None:
            raise ValueError("OpenAlex source is 'seed' but no `seed:` block is set")
        if self.source not in ('file', 'api', 'seed'):
            raise ValueError(
                f"Unknown OpenAlex source {self.source!r}; expected 'file', 'api' or 'seed'")

    @property
    def credentials(self) -> Union[None, OpenAlexApiConfig, OpenAlexSeedConfig]:
        """The block carrying the OpenAlex credentials, or None for a file
        source. Lets any consumer needing an OpenAlex client — the
        reference-key resolver, say — reuse the source's key and email
        whichever way the source was declared.

        Keyed on ``source`` rather than on whichever block happens to be filled
        in: a config file commonly keeps an unused block next to the active one
        (a ``file:`` path alongside ``api:``, say), and only the block naming
        the active source is parsed into a dataclass.
        """
        return {'api': self.api, 'seed': self.seed}.get(self.source)


@dataclass
class CleanConfig(ConfigField):
    min_signals_to_reject: int                    = 2
    extra_garbage_phrases: Union[None, List[str]] = None
    use_langdetect:        bool                   = False
    # Cap on the abstract length, in characters (None = keep whole). Sources
    # sometimes deliver a scraped page or a full text in the abstract field;
    # those records dominate the corpus' character count and the review bill.
    # The cut lands on a sentence boundary. 2500 keeps a full abstract.
    max_abstract_chars:    Union[None, int]       = None


@dataclass
class ExtractConfig(ConfigField):
    include_doc_type: Union[None, List[str]]         = None
    exclude_doc_type: Union[None, List[str]]         = None
    year:             int                            = 1900
    nb_citations:     int                            = 0
    language:         Union[None, str, List[str]]    = None
    scorer:           str                            = "partial_token_sort_ratio"
    score_cutoff:     int                            = 90


@dataclass
class ResolveReferencesConfig(ConfigField):
    enabled:            bool = False
    flag_unresolved:    bool = False
    fuzzy_score_cutoff: int  = 90
    ngram_size:         int  = 3
    max_candidates:     int  = 50
    scorer:             str  = "token_set_ratio"


@dataclass
class MergeConfig(ConfigField):
    title_similarity:       int = 98
    ngram_size:             int = 3
    max_candidates_per_row: int = 200
    scorer:                 str = "token_set_ratio"
    # Source priority for deduplication, most-important first. The first
    # configured source in this order becomes the merge's primary dataset: its
    # records win over duplicates from the others, its IDs are kept, and dropped
    # duplicates are aliased to the surviving record so references still resolve.
    # Default keeps OpenAlex first (stable IDs for coupling networks, abstracts
    # completed from WoS); listing WoS first restores the old WoS-wins behaviour.
    priority: List[str] = None

    def __post_init__(self):
        if self.priority is None:
            self.priority = ["open_alex", "wos", "scopus", "pubmed"]


@dataclass
class ReferenceKeysConfig(ConfigField):
    """Unify references onto a canonical DOI key space (coupling basis).

    Declared under ``bib.reference_keys``. Presence enables the step: each
    reference is remapped to its normalized DOI when derivable (native token
    kept otherwise), so bibliographic coupling holds across merged sources
    regardless of merge order.

    ``cache``
        Path to the persistent ``Wxxxx -> DOI`` CSV table. It accumulates across
        runs; ``None`` = compute in memory only, no persistence.
    ``resolve_external``
        When True, references that are opaque source ids pointing at extra-corpus
        works (whose DOI is not already known from the corpus rows or the cache)
        are resolved once to a DOI by that id scheme's resolver and cached. Each
        scheme uses its own credentials — OpenAlex (reused from ``open_alex.api``)
        is the only one implemented today; a scheme without credentials is
        skipped and its ids keep their native token. When False, only the free
        offline mapping runs. The resolution step is skipped automatically when
        the corpus has no DOI-bearing references from another source to bridge to
        (e.g. a pure-OpenAlex run), so the block is safe to leave enabled.
    """
    cache:            Union[None, str] = None
    resolve_external: bool             = True


@dataclass
class BibExportConfig(ConfigField):
    """Output configuration for the bib stage.

    Each run is stored in ``<export_dir>/<run_name>/bib_dataset.csv``.
    Call :meth:`resolve` (done automatically by ``BibDataset.from_config``)
    to finalise the run directory and set ``dataset`` on the instance.
    Leave ``run_name`` blank to auto-generate a timestamp.
    """
    export_dir: str
    run_name:   Union[None, str] = None
    dataset: str                 = None

    def resolve(self):
        """Create the run directory and set the output CSV path."""
        self.run_name, run_dir = _make_run_dir(self.export_dir, self.run_name)
        self.dataset = os.path.join(run_dir, 'bib_dataset.csv')
        return self


@dataclass
class BibConfig(ConfigField):
    wos:                Union[None, str, WosSourceConfig]          = None
    open_alex:          Union[None, str, OpenAlexSourceConfig]     = None
    scopus:             Union[None, str]                           = None
    pubmed:             Union[None, str]                           = None
    export:             Union[None, BibExportConfig]               = None
    clean:              CleanConfig                                = None
    extract:            ExtractConfig                              = None
    resolve_references: ResolveReferencesConfig                    = None
    merge:              MergeConfig                                = None
    reference_keys:      Union[None, ReferenceKeysConfig]              = None
    abstract_completion: Union[None, List[AbstractCompletionConfig]]   = None

    def __post_init__(self):
        super().__post_init__()
        # Backwards compatibility: a string under `wos:` / `open_alex:` is
        # treated as a file path. A dict is parsed as a structured source config.
        if isinstance(self.wos, dict):
            if not self.wos["source"]:
                self.wos = None
            else:
                self.wos = WosSourceConfig(**self.wos)
        elif isinstance(self.wos, str):
            self.wos = WosSourceConfig(source='file', file=self.wos)

        if isinstance(self.open_alex, dict):
            if not self.open_alex["source"]:
                self.open_alex = None
            else:
                self.open_alex = OpenAlexSourceConfig(**self.open_alex)
        elif isinstance(self.open_alex, str):
            self.open_alex = OpenAlexSourceConfig(source='file', file=self.open_alex)

        if isinstance(self.export, dict):
            self.export = BibExportConfig(**self.export)

        if isinstance(self.clean, dict):
            self.clean = CleanConfig(**self.clean)
        elif self.clean is None:
            self.clean = CleanConfig()

        if isinstance(self.extract, dict):
            self.extract = ExtractConfig(**self.extract)
        elif self.extract is None:
            self.extract = ExtractConfig()

        if isinstance(self.resolve_references, dict):
            self.resolve_references = ResolveReferencesConfig(**self.resolve_references)
        elif self.resolve_references is None:
            self.resolve_references = ResolveReferencesConfig()

        if isinstance(self.merge, dict):
            self.merge = MergeConfig(**self.merge)
        elif self.merge is None:
            self.merge = MergeConfig()

        # Presence of the block enables the canonical-key step; absence = None.
        if isinstance(self.reference_keys, dict):
            self.reference_keys = ReferenceKeysConfig(**self.reference_keys)

        # Corpus-level abstract completion: a list of provider steps run on the
        # merged corpus. Presence enables it; absence leaves it None.
        if self.abstract_completion is not None:
            self.abstract_completion = [
                c if isinstance(c, AbstractCompletionConfig)
                else AbstractCompletionConfig(**c)
                for c in self.abstract_completion
            ]


@dataclass
class HDBSCANConfig(ConfigField):
    # Topic granularity is driven by topic_model.desired_topics, not by raw
    # min_topic_size bounds: the ``min_topic_size`` grid is auto-derived from the
    # corpus size to aim at that band (see topic.derive_min_topic_size_range).
    # min_topic_size_range / topic_size_step are therefore internal (on
    # TopicModel), not user-facing.
    min_sample_range:         List[int] = field(default_factory=lambda: [2, 2])
    min_sample_step:          int       = 1
    cluster_selection_method: str       = 'leaf'
    metric:                   str       = 'euclidean'
    prediction_data:          bool      = True


@dataclass
class UMAPConfig(ConfigField):
    n_neighbors:  List[int] = field(default_factory=lambda: [5])
    n_components: List[int] = field(default_factory=lambda: [5])
    metric:       str       = 'cosine'
    min_dist:     float     = 0.0
    low_memory:   bool      = False
    random_state: int       = 42


@dataclass
class BerteleyConfig(ConfigField):
    allow_abbrev: bool = False


@dataclass
class CoherenceScorerConfig(ConfigField):
    ranking: str = "u_mass"
    purity:  str = "c_v"


@dataclass
class CTFIDFConfig(ConfigField):
    bm25_weighting:        bool = True
    reduce_frequent_words: bool = True


@dataclass
class TopicDistributionConfig(ConfigField):
    window:         int   = 8
    stride:         int   = 1
    min_similarity: float = 0.1
    batch_size:     int   = 1000


@dataclass
class BertopicConfig(ConfigField):
    transformer_model:       str  = 'allenai/specter2_base'
    n_gram_range:            str  = 'bigram'
    language:                str  = 'english'
    calculate_probabilities: bool = True


@dataclass
class ReviewExportConfig(ConfigField):
    """Output configuration for the review stage.

    Declare the parent directory (``export_dir``) and an optional run label
    (``run_name``).  If ``run_name`` is left blank, :meth:`resolve` generates
    a timestamp name (``YYYY-MM-DDTHHMMSS``) at run time so that successive
    test runs never overwrite each other.

    ``resolve()`` must be called before the review runs (done automatically by
    ``LLMReview.from_config``).  It creates the run directory, sets
    ``included_docs`` / ``total_docs`` on the instance, and defaults
    ``cache_dir`` to ``<run_dir>/cache/`` when not explicitly provided.

    Downstream sections (``bib_network``, ``topic_model``) can reference the
    output via ``config.review.export.included_docs`` after ``resolve()``,
    or leave ``doc_dataset`` blank to have ``Config.load`` auto-detect the
    most recent run.
    """
    export_dir: str
    run_name: str      = None  # None → auto-timestamp at resolve() time
    cache_dir: str     = None  # None → <run_dir>/cache/
    included_docs: str = None
    total_docs: str    = None

    def resolve(self):
        """Finalise run_name, create output directories, and set file paths."""
        self.run_name, run_dir = _make_run_dir(self.export_dir, self.run_name)
        self.included_docs = os.path.join(run_dir, 'reviewed_included.csv')
        self.total_docs    = os.path.join(run_dir, 'reviewed_total.csv')
        if self.cache_dir is None:
            self.cache_dir = os.path.join(run_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        return self


@dataclass
class ReviewerConfig(ConfigField):
    """Mirror of one entry under `review.reviewers` in the YAML.
    Cross-section fields like inclusion_criteria are NOT here — they live
    at the ReviewConfig level and are wired together by the runtime layer."""
    model_id:                str
    host:                    str
    provider:                str
    name:                    str
    max_tokens:              int
    temperature:             float
    reasoning_effort:        str
    backstory:               str
    additional_context:      str
    reasoning:               str              = 'brief'
    max_retries:             Union[None, int] = None  # None → falls back to ReviewConfig value
    max_concurrent_requests: Union[None, int] = None  # None → falls back to ReviewConfig value
    items_per_call:          Union[None, int] = None  # None → falls back to ReviewConfig value
    requests_per_minute:     Union[None, int] = None  # None → falls back to ReviewConfig value,
                                                      # then to the provider default (Albert: 10)


@dataclass
class ReviewConfig(ConfigField):
    # --- Required ---
    export:               ReviewExportConfig
    text_inputs:          List[str]
    inclusion_criteria:   str
    exclusion_criteria:   str
    reviewers:            List[ReviewerConfig]
    workflow:             List[dict]
    # --- Optional (section-level defaults) ---
    doc_dataset:             Union[None, str] = None          # None = auto-detect latest bib run
    batch_size:              int              = 100
    api_pause:               float            = 30.0
    decision_rule:           str              = 'majority'    # majority | mean
    sample_size:             Union[None, int] = None          # None = process full dataset
    sample_seed:             Union[None, int] = None          # seed for that draw; None = a new sample every run
    max_retries:             Union[None, int] = None          # None → module default (2);  overridable per reviewer
    max_concurrent_requests: Union[None, int] = None          # None → module default (10); overridable per reviewer
    items_per_call:          Union[None, int] = None          # None → module default (1);  overridable per reviewer
    requests_per_minute:     Union[None, int] = None          # None → provider default (Albert: 10, others: unpaced)
    # Deferred-batch transport: same prompts, submitted in bulk and collected
    # within 24 h, at half price. Requires a provider that offers such an
    # endpoint (currently anthropic). Off by default — it trades latency for
    # money, and that is the user's call, not a default.
    use_batch_api:           bool             = False
    batch_poll_interval:     float            = 30.0          # seconds between status checks

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.export, dict):
            self.export = ReviewExportConfig(**self.export)
        self.reviewers = [ReviewerConfig(**r) for r in self.reviewers]


@dataclass
class TopicExportConfig(ConfigField):
    """Output configuration for the topic-model stage.

    Each run is stored in its own sub-directory: ``<export_dir>/<run_name>/``.
    Leave ``run_name`` blank to auto-generate a timestamp at run time
    (directory creation is deferred to ``TopicModel.run()``).
    """
    export_dir: str
    run_name:   Union[None, str] = None


@dataclass
class TopicModelConfig(ConfigField):
    export:              TopicExportConfig
    doc_dataset:         Union[None, str]                       = None
    distance:            str                                    = "euclidean"
    keep_n_results:      int                                    = 10
    best_model_index:    int                                    = 0
    desired_topics:      List[int]                              = field(
        default_factory=lambda: [5, 15])
    coherence_scorer:    CoherenceScorerConfig                  = None
    hdbscan:             HDBSCANConfig                          = None
    umap:                UMAPConfig                             = None
    bertopic:            BertopicConfig                         = None
    berteley:            BerteleyConfig                         = None
    ctfidf:              CTFIDFConfig                           = None
    topic_distribution:  TopicDistributionConfig                = None

    def __post_init__(self):
        super().__post_init__()

        # Desired topic-count band [min, max]: the sole user-facing knob for topic
        # granularity. The HDBSCAN min_topic_size grid is auto-derived from the
        # corpus size to aim at it, and only models whose topic count lands in the
        # band are kept (see topic.derive_min_topic_size_range / topic_modeling).
        band = list(self.desired_topics)
        if len(band) != 2 or band[0] < 1 or band[1] < band[0]:
            raise ValueError(
                f"desired_topics must be [min, max] with 1 <= min <= max; "
                f"got {self.desired_topics!r}")
        self.desired_topics = band

        if isinstance(self.export, dict):
            self.export = TopicExportConfig(**self.export)

        if isinstance(self.hdbscan, dict):
            self.hdbscan = HDBSCANConfig(**self.hdbscan)
        elif self.hdbscan is None:
            self.hdbscan = HDBSCANConfig()

        if isinstance(self.umap, dict):
            self.umap = UMAPConfig(**self.umap)
        elif self.umap is None:
            self.umap = UMAPConfig()

        if isinstance(self.bertopic, dict):
            self.bertopic = BertopicConfig(**self.bertopic)
        elif self.bertopic is None:
            self.bertopic = BertopicConfig()

        if isinstance(self.berteley, dict):
            self.berteley = BerteleyConfig(**self.berteley)
        elif self.berteley is None:
            self.berteley = BerteleyConfig()

        if isinstance(self.ctfidf, dict):
            self.ctfidf = CTFIDFConfig(**self.ctfidf)
        elif self.ctfidf is None:
            self.ctfidf = CTFIDFConfig()

        if isinstance(self.topic_distribution, dict):
            self.topic_distribution = TopicDistributionConfig(**self.topic_distribution)
        elif self.topic_distribution is None:
            self.topic_distribution = TopicDistributionConfig()

        if isinstance(self.coherence_scorer, dict):
            self.coherence_scorer = CoherenceScorerConfig(**self.coherence_scorer)
        elif self.coherence_scorer is None:
            self.coherence_scorer = CoherenceScorerConfig()


@dataclass
class TopicsSectionConfig(ConfigField):
    n_repr_docs_per_topic: int = 5


@dataclass
class CouplingPanelConfig(ConfigField):
    """Coupling network panel (document × document)."""
    resolution_range: List[float] = field(default_factory=lambda: [0.4, 1.2])  # [min, max]: sweep and keep the best-modularity partition
    resolution_step:  float = 0.1  # step of the resolution sweep
    min_size:   int   = 5          # communities smaller than this → uncoupled tail
    backbone_k: int   = 3          # draw only each node's k strongest couplings
    color_by:   str   = "topic"    # "topic" | "community"


@dataclass
class ReferenceMetadataConfig(ConfigField):
    """Resolve the co-citation nodes back to real works (title, year, authors…).

    Declared under ``…bib_network.cocitation.metadata``; its presence enables the
    step. A co-citation node is a *reference*, usually outside the corpus, so
    nothing local knows what it is: without this, the panel's clusters cannot be
    named and its hover shows only an identifier. With it, each community gets
    TF-IDF sub-themes, an LLM label, and a most-cited-authors profile — the same
    treatment the coupling communities already get from their documents' text.

    ``provider`` selects the backend: ``openalex`` is the only one implemented
    today; another metadata source would register its own fetcher the same way.
    ``api_key`` / ``email`` are that provider's credentials (for OpenAlex the
    email is the polite pool, optional but recommended). The report stage carries
    its own credentials rather than reusing ``bib.open_alex``: a report is
    commonly re-run on its own, long after the corpus was built.

    ``cache``
        Persistent ``reference_key -> metadata`` CSV, accumulated across runs;
        ``None`` = fetch every run, no persistence.
    ``include_abstracts``
        Also pull the references' abstracts — richer TF-IDF terms, at a much
        larger payload and cache. Turning it on does not back-fill abstracts for
        already-cached references; delete ``cache`` to refetch them.
    """
    provider:          str = 'openalex'      # 'openalex' today
    api_key:           Union[None, str] = None   # typically `${OPENALEX_API_KEY}`
    email:             Union[None, str] = None   # polite pool
    cache:             Union[None, str] = None
    include_abstracts: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            raise ValueError(
                f"cocitation.metadata provider {self.provider!r} needs an `api_key:`")


@dataclass
class CocitationPanelConfig(ConfigField):
    """Co-citation network panel (reference × reference)."""
    min_ref_freq: int   = 2         # keep references cited by at least this many documents
    resolution_range: List[float] = field(default_factory=lambda: [0.4, 1.2])  # [min, max]: sweep and keep the best-modularity partition
    resolution_step:  float = 0.1   # step of the resolution sweep
    min_size:     int   = 5         # communities smaller than this → tail
    backbone_k:   int   = 3         # draw only each node's k strongest co-citations
    color_by:     str   = "community"  # "topic" | "community"
    metadata:     Union[None, ReferenceMetadataConfig] = None  # resolve node ids to real works

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.metadata, dict):
            self.metadata = ReferenceMetadataConfig(**self.metadata)


@dataclass
class ConnectivityConfig(ConfigField):
    """Inter-group connectivity matrix panels (rendered for BOTH topic and
    community groupings)."""
    by:    str   = "topic"          # deprecated / ignored — both topic and community panels are rendered
    scale: float = 1000.0           # rescales the tiny coupling values for readability


@dataclass
class BibNetworkSectionConfig(ConfigField):
    """Bibliographic-network panels of the report.

    The coupling and co-citation networks are recomputed from the reviewed
    dataset's raw references (Salton / co-citation matrix → Leiden communities →
    backbone layout). Coupling is coloured by BERTopic topic to reveal the topic
    mix of each community; co-citation by Leiden community by default. Like every
    other report section, the panels render whenever their prerequisite is met —
    here, a reviewed dataset carrying a ``references`` column.
    """
    coupling:     CouplingPanelConfig   = None
    cocitation:   CocitationPanelConfig = None
    connectivity: ConnectivityConfig    = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.coupling, dict):
            self.coupling = CouplingPanelConfig(**self.coupling)
        elif self.coupling is None:
            self.coupling = CouplingPanelConfig()
        if isinstance(self.cocitation, dict):
            self.cocitation = CocitationPanelConfig(**self.cocitation)
        elif self.cocitation is None:
            self.cocitation = CocitationPanelConfig()
        if isinstance(self.connectivity, dict):
            self.connectivity = ConnectivityConfig(**self.connectivity)
        elif self.connectivity is None:
            self.connectivity = ConnectivityConfig()


@dataclass
class TemporalSectionConfig(ConfigField):
    variants: List[str] = field(default_factory=lambda: [
        "absolute", "normalized", "weighted"
    ])


@dataclass
class TopicCharacteristicsConfig(ConfigField):
    n_top_cited_per_topic: int = 5
    n_top_cited_global:    int = 50


@dataclass
class TopicSimilarityConfig(ConfigField):
    clustering: bool = True
    dendrogram: bool = True


@dataclass
class CompositeConfig(ConfigField):
    """Tuning of the three-axis composite selector (selection_by: composite)."""
    aggregate:         str  = "mean"    # "mean" | "gmean" | "chebyshev"
    weights:           Union[None, List[float]] = None  # (centrality, representativeness, relevance); None = equal
    relevance_mode:    str  = "blend"   # "blend" (cit/yr + raw citations) | "cpy" (cit/yr alone)
    drop_current_year: bool = True      # drop the incomplete current year from the cit/yr denominator
    current_year:      Union[None, int] = None  # reference year; None = latest year in the corpus
    split_current_year: bool = False    # separate current-year "research fronts" (2-axis) from the historical 3-axis list

    def __post_init__(self):
        super().__post_init__()
        if self.weights is not None and len(self.weights) != 3:
            raise ValueError(
                "paper_selection.composite.weights must have exactly 3 values "
                "(centrality, representativeness, relevance)"
            )


@dataclass
class PaperSelectionConfig(ConfigField):
    min_year:             int   = 2000
    proportion_per_topic: float = 0.15
    # "citations" | "random" | "coupling" | "co_citation" | "composite"
    # "composite" ranks papers per topic by the three-axis indicator (coupling
    # centrality + citation relevance + thematic representativeness); tune it
    # under the `composite:` sub-block.
    selection_by:         str   = "citations"
    composite:            CompositeConfig = None
    export_annex:         bool  = True
    annex_format:         str   = "csv"   # "csv" | "txt"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.composite, dict):
            self.composite = CompositeConfig(**self.composite)
        elif self.composite is None:
            self.composite = CompositeConfig()


@dataclass
class ReportSectionsConfig(ConfigField):
    topics:                TopicsSectionConfig       = None
    bib_network:           BibNetworkSectionConfig   = None
    temporal:              TemporalSectionConfig     = None
    topic_characteristics: TopicCharacteristicsConfig = None
    topic_similarity:      TopicSimilarityConfig     = None
    paper_selection:       PaperSelectionConfig      = None
    extra:                 Union[None, List[dict]]   = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.topics, dict):
            self.topics = TopicsSectionConfig(**self.topics)
        elif self.topics is None:
            self.topics = TopicsSectionConfig()

        if isinstance(self.bib_network, dict):
            self.bib_network = BibNetworkSectionConfig(**self.bib_network)
        elif self.bib_network is None:
            self.bib_network = BibNetworkSectionConfig()

        if isinstance(self.temporal, dict):
            self.temporal = TemporalSectionConfig(**self.temporal)
        elif self.temporal is None:
            self.temporal = TemporalSectionConfig()

        if isinstance(self.topic_characteristics, dict):
            self.topic_characteristics = TopicCharacteristicsConfig(**self.topic_characteristics)
        elif self.topic_characteristics is None:
            self.topic_characteristics = TopicCharacteristicsConfig()

        if isinstance(self.topic_similarity, dict):
            self.topic_similarity = TopicSimilarityConfig(**self.topic_similarity)
        elif self.topic_similarity is None:
            self.topic_similarity = TopicSimilarityConfig()

        if isinstance(self.paper_selection, dict):
            self.paper_selection = PaperSelectionConfig(**self.paper_selection)
        elif self.paper_selection is None:
            self.paper_selection = PaperSelectionConfig()


@dataclass
class ReportMetaConfig(ConfigField):
    title:       str              = "Bibliographic report — Pysyrev"
    subtitle:    Union[None, str] = None
    author:      str              = "Report generated with the pysyrev engine (v0.1)"
    date_format: str              = "%d/%m/%Y"
    version:     str              = "1.0.0"
    summary:     Union[None, str] = None


@dataclass
class TopicLabelerConfig(ConfigField):
    """LLM configuration for generating human-readable topic labels."""
    provider:                str
    model_id:                str
    host:                    Union[None, str] = None
    max_tokens:              int              = 200
    temperature:             float            = 0.3
    max_retries:             int              = 2
    max_concurrent_requests: int              = 5
    n_repr_docs_for_labeling: int             = 3
    system_prompt:           Union[None, str] = None


@dataclass
class TopicReportConfig(ConfigField):
    """PDF layout parameters for the topic-report stage."""
    run_dir:     str = None  # auto-detected by Config.load() from topic_model.export.export_dir when blank
    export_to:   str = None
    meta:        Union[None, ReportMetaConfig]     = None
    sections:    Union[None, ReportSectionsConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.meta, dict):
            self.meta = ReportMetaConfig(**self.meta)
        elif self.meta is None:
            self.meta = ReportMetaConfig()
        if isinstance(self.sections, dict):
            self.sections = ReportSectionsConfig(**self.sections)
        elif self.sections is None:
            self.sections = ReportSectionsConfig()


@dataclass
class UnpaywallConfig(ConfigField):
    """Unpaywall source for open-access PDF discovery.

    `email` is required by Unpaywall's politeness policy and unlocks the
    higher-throughput pool.  See https://unpaywall.org/products/api.
    """
    email: str


@dataclass
class ElsevierDownloadConfig(ConfigField):
    """Elsevier TDM (Text and Data Mining) source.

    Requires an institutional API key.  When `format` is ``xml``, the
    full-text XML is retrieved (preferred for downstream LLM use because
    the text is already structured).  Use ``pdf`` to retrieve the raw PDF.
    See https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl.
    """
    api_key: str
    format:  str = 'xml'  # 'xml' | 'pdf'

    def __post_init__(self):
        super().__post_init__()
        if self.format not in ('xml', 'pdf'):
            raise ValueError(
                f"elsevier.format must be 'xml' or 'pdf', got {self.format!r}"
            )


@dataclass
class DownloadConfig(ConfigField):
    """Standalone configuration for the PDF/full-text download module.

    Loaded from a dedicated YAML file (separate from the main pipeline
    config) via :meth:`DownloadConfig.load`.

    Cascade order for each paper:
      1. Unpaywall  — finds a legal open-access PDF by DOI.
      2. OpenAlex   — uses ``open_access.oa_url`` already present in the
                      input dataset when the bib stage has been run.
      3. Elsevier   — TDM API, only when an ``elsevier.api_key`` is set.

    Papers that cannot be retrieved are listed in ``download_report.csv``
    with ``status = manual`` so the user knows what still needs attention.
    """
    doc_dataset:   str                              # path to input CSV (must have a `doi` column)
    output_dir:    str                              # root directory for all download output
    format:        str                = 'pdf'       # default format for OA sources: 'pdf' | 'xml'
    max_papers:    Union[None, int]   = None        # cap the number of attempted downloads
    request_delay: float              = 1.0         # seconds between HTTP requests (Unpaywall: ≤ 1 req/s)
    unpaywall:     Union[None, UnpaywallConfig]      = None
    elsevier:      Union[None, ElsevierDownloadConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.format not in ('pdf', 'xml'):
            raise ValueError(
                f"download.format must be 'pdf' or 'xml', got {self.format!r}"
            )
        if isinstance(self.unpaywall, dict):
            self.unpaywall = UnpaywallConfig(**self.unpaywall)
        if isinstance(self.elsevier, dict):
            self.elsevier = ElsevierDownloadConfig(**self.elsevier)
        os.makedirs(os.path.join(self.output_dir, 'papers'), exist_ok=True)

    @classmethod
    def load(cls, config_file: str) -> 'DownloadConfig':
        """Load a standalone download YAML config file.

        Steps:
          1. Read the YAML.
          2. Load the .env file referenced by the root-level ``env:`` key (if any).
          3. Resolve all ``${VAR}`` references.
          4. Build the typed dataclass.
        """
        with open(config_file, 'r') as fh:
            raw = yaml.safe_load(fh) or {}

        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        resolved = _resolve_env_vars(raw)
        resolved.pop('env', None)
        return cls(**resolved)


@dataclass
class Config:
    """Root configuration object.

    All sections are optional — only the sections present in the YAML are
    executed. The canonical stage order is:
    ``bib → review → topic-model → topic-report``.

    ``Config.load()`` propagates outputs between stages automatically when
    ``doc_dataset`` / ``run_dir`` are left blank, so a full-pipeline YAML
    requires no explicit cross-section paths.
    """
    env:                Union[None, str]                    = None
    bib:                Union[None, BibConfig]              = None
    review:             Union[None, ReviewConfig]           = None
    topic_model:        Union[None, TopicModelConfig]       = None
    topic_report:       Union[None, TopicReportConfig]      = None
    llm:                Union[None, TopicLabelerConfig]     = None

    @classmethod
    def load(cls, config_file):
        """Load a YAML config file.

        Steps:
          1. Read the YAML.
          2. Load the .env file referenced by the root-level ``env:`` key (if any).
          3. Resolve all ``${VAR}`` references.
          4. Propagate outputs between stages when ``doc_dataset`` / ``run_dir``
             are left blank (auto-detection of the latest run in each export_dir).
          5. Build typed dataclasses for every section present.
        """
        with open(config_file, 'r') as file:
            raw = yaml.safe_load(file) or {}

        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        resolved = _resolve_env_vars(raw)

        # ── Propagate outputs between stages when doc_dataset is blank ────────
        #   bib.export.export_dir    → review.doc_dataset
        #   review.export.export_dir → topic_model.doc_dataset
        #     (topic_model.doc_dataset — the reviewed_included.csv — is also the
        #      source the report's network panels are recomputed from.)
        bib_export_dir    = (resolved.get('bib') or {}).get('export', {}).get('export_dir')
        review_export_dir = (resolved.get('review') or {}).get('export', {}).get('export_dir')
        review_data      = dict(resolved.get('review') or {})
        topic_model_data = dict(resolved.get('topic_model') or {})

        if bib_export_dir and not review_data.get('doc_dataset'):
            latest = _find_latest_file(bib_export_dir, 'bib_dataset.csv')
            if latest:
                review_data['doc_dataset'] = latest

        if review_export_dir and not topic_model_data.get('doc_dataset'):
            latest = _find_latest_file(review_export_dir, 'reviewed_included.csv')
            if latest:
                topic_model_data['doc_dataset'] = latest

        # ── Auto-detect topic_report.run_dir from latest topic_model run ─────
        topic_report_data = dict(resolved.get('topic_report') or {})
        if topic_report_data and not topic_report_data.get('run_dir'):
            tm_export_dir = (
                topic_model_data.get('export', {}).get('export_dir')
                or (resolved.get('topic_model') or {}).get('export', {}).get('export_dir')
            )
            if tm_export_dir:
                latest = _find_latest_dir(tm_export_dir)
                if latest:
                    topic_report_data['run_dir'] = latest

        return cls(
            env                = resolved.get('env'),
            bib                = BibConfig(**resolved['bib'])            if resolved.get('bib')          else None,
            review             = ReviewConfig(**review_data)             if review_data                   else None,
            topic_model        = TopicModelConfig(**topic_model_data)    if topic_model_data              else None,
            topic_report       = TopicReportConfig(**topic_report_data)  if topic_report_data             else None,
            llm                = TopicLabelerConfig(**resolved['llm'])   if resolved.get('llm')           else None,
        )
