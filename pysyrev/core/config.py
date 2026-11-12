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

Validation is per stage: loading only records what is wrong with a section
(see :attr:`Config.errors`), so a section no stage asked for cannot stop a run.
The error is raised when a stage that needs that section is run.
"""
import dataclasses
import datetime
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Set, Union, get_args, get_origin

import yaml
from dotenv import load_dotenv


# Pattern to resolve ${ENV_VAR} references inside YAML string values.
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


# =============================================================================
# Config errors, dotted paths, and mandatory keys
# =============================================================================

class ConfigError(ValueError):
    """A bad config node, named by its dotted YAML path (``review.export``)."""

    def __init__(self, message: str, path: str = ''):
        self.path = path
        self.section = path.split('.')[0].split('[')[0] if path else ''
        super().__init__(message)


#: Dotted path of the node currently being built. Loading is synchronous, so a
#: module-level stack is enough for a sub-block to name itself.
_PATH: List[str] = []


@contextmanager
def _at(key: str):
    _PATH.append(key)
    try:
        yield
    finally:
        _PATH.pop()


def _dotted(*extra: str) -> str:
    """The dotted path of the node being built, plus *extra* keys under it."""
    return '.'.join([*_PATH, *extra])


def _is_mandatory(field_) -> bool:
    """True for a field the YAML must fill: no default, no default factory."""
    return (isinstance(field_.default, dataclasses._MISSING_TYPE)
            and isinstance(field_.default_factory, dataclasses._MISSING_TYPE))


def _instantiate(cls, block: dict):
    """Build *cls* from the YAML mapping *block*, naming bad keys by path."""
    renamed = getattr(cls, 'RENAMED_KEYS', {})
    block = {renamed.get(k, k): v for k, v in block.items()}

    known = {f.name for f in fields(cls)}
    unknown = sorted(k for k in block if k not in known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {', '.join(_dotted(k) for k in unknown)}. "
            f"Keys accepted here: {', '.join(sorted(known))}.",
            path=_dotted())

    missing = [f.name for f in fields(cls)
               if _is_mandatory(f) and block.get(f.name) is None]
    if missing:
        raise ConfigError(
            f"mandatory key(s) missing or left empty: "
            f"{', '.join(_dotted(k) for k in missing)}",
            path=_dotted())

    # A key written but left blank (`reasoning:`) reads as None in YAML, which
    # would override the field's default rather than fall back to it. Blank
    # means "use the default", as the template's comments promise.
    optional = {f.name for f in fields(cls) if not _is_mandatory(f)}
    block = {k: v for k, v in block.items() if not (v is None and k in optional)}

    return cls(**block)


def _build(cls, value, key: str):
    """Parse the sub-block under *key* into *cls*; anything already parsed, or
    blank, is returned untouched."""
    if not isinstance(value, dict):
        return value
    with _at(key):
        return _instantiate(cls, value)


def _with_defaults(cls, value, key: str):
    """Like :func:`_build`, but an absent block falls back to *cls*'s defaults."""
    built = _build(cls, value, key)
    return cls() if built is None else built


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


#: Sections that write results, in stage order. Under a root-level
#: ``output_dir``, each one exports to ``<output_dir>/<section>/`` by default.
OUTPUT_SECTIONS = ('bib', 'review', 'topic_model', 'topic_report')


def _resolve_export_dirs(resolved: dict) -> dict:
    """Fill every section's ``export.export_dir`` from the root ``output_dir``.

    - blank ``export_dir``    → ``<output_dir>/<section>``
    - relative ``export_dir`` → ``<output_dir>/<export_dir>``
    - absolute ``export_dir`` → kept as is (an explicit override)

    Without ``output_dir`` the sections are left as written. The legacy flat
    ``topic_report.export_to`` is folded into ``topic_report.export.export_dir``.
    Returns a new mapping; *resolved* is not modified.
    """
    out = dict(resolved)
    output_dir = out.get('output_dir')
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        out['output_dir'] = output_dir

    report = out.get('topic_report')
    if isinstance(report, dict) and 'export_to' in report:
        print("[config] `topic_report.export_to` is deprecated — use "
              "`topic_report.export.export_dir`, or a root-level `output_dir`.")
        report = dict(report)
        legacy = report.pop('export_to')
        export = dict(report.get('export') or {})
        if legacy and not export.get('export_dir'):
            export['export_dir'] = legacy
        if export:
            report['export'] = export
        out['topic_report'] = report

    if not output_dir:
        return out

    for name in OUTPUT_SECTIONS:
        data = out.get(name)
        if not isinstance(data, dict) or not data:
            continue
        export = data.get('export')
        if export is not None and not isinstance(export, dict):
            continue  # malformed: let the section's own parsing name it
        export = dict(export or {})
        export_dir = export.get('export_dir')
        if not export_dir:
            export['export_dir'] = os.path.join(output_dir, name)
        elif not os.path.isabs(os.path.expanduser(export_dir)):
            export['export_dir'] = os.path.join(output_dir, export_dir)
        out[name] = {**data, 'export': export}
    return out


def _export_dir_of(resolved: dict, section: str) -> Union[None, str]:
    """Where *section* exports: its own ``export.export_dir`` when declared,
    else ``<output_dir>/<section>`` — so a stage finds its input under the
    project root even when the upstream section is absent from this config."""
    export = (resolved.get(section) or {}).get('export')
    if isinstance(export, dict) and export.get('export_dir'):
        return export['export_dir']
    output_dir = resolved.get('output_dir')
    return os.path.join(output_dir, section) if output_dir else None



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

    #: Keys mandatory only under some `source:` — reported by `pysyrev check`.
    CONDITIONAL = ("`file:` when source is 'file'", "`api:` when source is 'api'")

    def __post_init__(self):
        super().__post_init__()
        if self.source == 'api':
            self.api = _build(WosApiConfig, self.api, 'api')
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

    CONDITIONAL = ("`provider:` and `api_key:` in every entry",)

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

    CONDITIONAL = ("`file:` or `seeds:` — at least one of the two",)

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

    #: Keys mandatory only under some `source:` — reported by `pysyrev check`.
    CONDITIONAL = ("`file:` when source is 'file'", "`api:` when source is 'api'",
                   "`seed:` when source is 'seed'")

    def __post_init__(self):
        super().__post_init__()
        if self.source == 'api':
            self.api = _build(OpenAlexApiConfig, self.api, 'api')
        if self.source == 'seed':
            self.seed = _build(OpenAlexSeedConfig, self.seed, 'seed')
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
    # Drop a record left without an abstract — none delivered by any source, or
    # one the filter above rejected. True is the right default for a
    # search-built corpus: a record with no abstract cannot be screened. Set it
    # to False for a closed corpus of known papers (a DOI list), where silently
    # losing a record matters more than carrying one the reviewer cannot read.
    drop_record_with_no_abstract: bool            = True


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

    CONDITIONAL = ("at least one source among `wos:`, `open_alex:`, `scopus:`, `pubmed:`",)

    def __post_init__(self):
        super().__post_init__()
        # Backwards compatibility: a string under `wos:` / `open_alex:` is
        # treated as a file path. A dict is parsed as a structured source config.
        # A blank `source:` is how a config file parks a source it is not using.
        for name, source_cls in (('wos', WosSourceConfig),
                                 ('open_alex', OpenAlexSourceConfig)):
            value = getattr(self, name)
            if isinstance(value, str):
                setattr(self, name, source_cls(source='file', file=value))
            elif isinstance(value, dict) and 'source' in value and not value['source']:
                setattr(self, name, None)
            else:
                setattr(self, name, _build(source_cls, value, name))

        self.export = _build(BibExportConfig, self.export, 'export')

        self.clean   = _with_defaults(CleanConfig, self.clean, 'clean')
        self.extract = _with_defaults(ExtractConfig, self.extract, 'extract')
        self.merge   = _with_defaults(MergeConfig, self.merge, 'merge')
        self.resolve_references = _with_defaults(
            ResolveReferencesConfig, self.resolve_references, 'resolve_references')

        # Presence of the block enables the canonical-key step; absence = None.
        self.reference_keys = _build(
            ReferenceKeysConfig, self.reference_keys, 'reference_keys')

        # Corpus-level abstract completion: a list of provider steps run on the
        # merged corpus. Presence enables it; absence leaves it None.
        if self.abstract_completion is not None:
            self.abstract_completion = [
                _build(AbstractCompletionConfig, c, f'abstract_completion[{i}]')
                for i, c in enumerate(self.abstract_completion)
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
    provider:                str
    name:                    str
    backstory:               str
    # Every model knob below is routinely left blank, and the runtime simply
    # omits the ones it is not given: a hosted provider needs no host, and
    # max_tokens / temperature / reasoning_effort are not offered by every model.
    host:                    Union[None, str]   = None
    max_tokens:              Union[None, int]   = None
    temperature:             Union[None, float] = None
    reasoning_effort:        Union[None, str]   = None
    additional_context:      Union[None, str]   = None
    reasoning:               str                = 'brief'
    max_retries:             Union[None, int] = None  # None → falls back to ReviewConfig value
    max_concurrent_requests: Union[None, int] = None  # None → falls back to ReviewConfig value
    items_per_call:          Union[None, int] = None  # None → falls back to ReviewConfig value
    requests_per_minute:     Union[None, int] = None  # None → falls back to ReviewConfig value,
                                                      # then to the provider default (Albert: 10)
    timeout:                 Union[None, float] = None  # seconds per call; None → ReviewConfig
                                                        # value, then the provider default


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
    timeout:                 Union[None, float] = None        # seconds per call; None → provider default
                                                              # (Albert: 120; openai/anthropic: 600; litellm: 6000)
    # Deferred-batch transport: same prompts, submitted in bulk and collected
    # within 24 h, at half price. Requires a provider that offers such an
    # endpoint (currently anthropic). Off by default — it trades latency for
    # money, and that is the user's call, not a default.
    use_batch_api:           bool             = False
    batch_poll_interval:     float            = 30.0          # seconds between status checks

    def __post_init__(self):
        super().__post_init__()
        self.export = _build(ReviewExportConfig, self.export, 'export')
        self.reviewers = [_build(ReviewerConfig, r, f'reviewers[{i}]')
                          for i, r in enumerate(self.reviewers)]


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

        self.export   = _build(TopicExportConfig, self.export, 'export')
        self.hdbscan  = _with_defaults(HDBSCANConfig, self.hdbscan, 'hdbscan')
        self.umap     = _with_defaults(UMAPConfig, self.umap, 'umap')
        self.bertopic = _with_defaults(BertopicConfig, self.bertopic, 'bertopic')
        self.berteley = _with_defaults(BerteleyConfig, self.berteley, 'berteley')
        self.ctfidf   = _with_defaults(CTFIDFConfig, self.ctfidf, 'ctfidf')
        self.topic_distribution = _with_defaults(
            TopicDistributionConfig, self.topic_distribution, 'topic_distribution')
        self.coherence_scorer = _with_defaults(
            CoherenceScorerConfig, self.coherence_scorer, 'coherence_scorer')


@dataclass
class TopicsSectionConfig(ConfigField):
    n_repr_docs_per_topic: int = 5


def _check_edge_style(cfg, where: str) -> None:
    """Validate a panel's ``edge_scale`` / ``edge_width_range``."""
    if cfg.edge_scale not in ("tertile", "linear"):
        raise ValueError(f"{where}.edge_scale must be 'tertile' or 'linear', "
                         f"got {cfg.edge_scale!r}")
    r = cfg.edge_width_range
    if len(r) != 2 or not 0 < r[0] < r[1]:
        raise ValueError(f"{where}.edge_width_range must be [min, max] with 0 < min < max")


@dataclass
class CouplingPanelConfig(ConfigField):
    """Coupling network panel (document × document).

    Edge width always shows the Salton-normalised coupling (raw shared
    references would favour reference-rich reviews); ``edge_scale`` picks how:
    ``tertile`` (three widths by rank, the default) or ``linear`` (width
    proportional to the strength, over ``edge_width_range`` px). The raw
    number of shared references is in each edge's hover (HTML export).
    """
    resolution_range: List[float] = field(default_factory=lambda: [0.4, 1.2])  # [min, max]: sweep and keep the best-modularity partition
    resolution_step:  float = 0.1  # step of the resolution sweep
    min_size:   int   = 5          # communities smaller than this → uncoupled tail
    backbone_k: int   = 3          # draw only each node's k strongest couplings
    color_by:   str   = "topic"    # "topic" | "community"
    edge_scale: str   = "tertile"  # "tertile" | "linear"
    edge_width_range: List[float] = field(default_factory=lambda: [0.4, 3.5])  # linear scale only

    def __post_init__(self):
        super().__post_init__()
        _check_edge_style(self, "bib_network.coupling")


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

    CONDITIONAL = ("`api_key:` once the block is declared",)

    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            raise ValueError(
                f"cocitation.metadata provider {self.provider!r} needs an `api_key:`")


@dataclass
class CocitationPanelConfig(ConfigField):
    """Co-citation network panel (reference × reference).

    Edge width always shows the Salton-normalised co-citation (raw counts
    would link any two widely cited classics); ``edge_scale`` /
    ``edge_width_range`` as for :class:`CouplingPanelConfig`. The raw number
    of documents citing both references is in each edge's hover.
    """
    min_ref_freq: int   = 2         # keep references cited by at least this many documents
    resolution_range: List[float] = field(default_factory=lambda: [0.4, 1.2])  # [min, max]: sweep and keep the best-modularity partition
    resolution_step:  float = 0.1   # step of the resolution sweep
    min_size:     int   = 5         # communities smaller than this → tail
    backbone_k:   int   = 3         # draw only each node's k strongest co-citations
    color_by:     str   = "community"  # "topic" | "community"
    metadata:     Union[None, ReferenceMetadataConfig] = None  # resolve node ids to real works
    edge_scale:   str   = "tertile"    # "tertile" | "linear"
    edge_width_range: List[float] = field(default_factory=lambda: [0.4, 3.5])  # linear scale only

    def __post_init__(self):
        super().__post_init__()
        _check_edge_style(self, "bib_network.cocitation")
        self.metadata = _build(ReferenceMetadataConfig, self.metadata, 'metadata')


@dataclass
class ConnectivityConfig(ConfigField):
    """Inter-group connectivity matrix panels (rendered for BOTH topic and
    community groupings).

    Blocks are scored as observed/expected against a degree-preserving null, a
    ratio whose baseline is 1 and which therefore needs no rescaling — *by* and
    *scale* are inert, kept so existing configs still load.
    """
    by:    str   = "topic"          # deprecated / ignored — both topic and community panels are rendered
    scale: float = 1000.0           # deprecated / ignored — the obs/exp ratio needs no rescaling

    # Observations required before a group's internal ratio is reported at all.
    # Below it the cell is left blank rather than filled with a number no amount
    # of evidence backs: a ratio computed from two or three observations is not
    # a small ratio, it is an unknown one. Raise it on a large corpus, where a
    # community backed by ten documents is still thin.
    min_support: int = 10


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
        self.coupling     = _with_defaults(CouplingPanelConfig, self.coupling, 'coupling')
        self.cocitation   = _with_defaults(CocitationPanelConfig, self.cocitation, 'cocitation')
        self.connectivity = _with_defaults(ConnectivityConfig, self.connectivity, 'connectivity')


@dataclass
class ActorNetworkConfig(ConfigField):
    """Collaboration network of one actor level (``…actors.network``).

    Two actors are linked by the papers they share; the strongest
    *max_nodes* actors with at least *min_papers* papers are drawn (the
    matrix is dense, and an author network past a few hundred nodes is
    unreadable anyway). ``normalize``: ``salton`` (shared / sqrt(p_i p_j), as
    the coupling network), ``association`` (observed / expected, VOSviewer's
    association strength) or ``count`` (raw shared papers).
    """
    enabled:          bool  = True
    min_papers:       int   = 2
    max_nodes:        int   = 150
    normalize:        str   = "salton"   # "salton" | "association" | "count"
    resolution_range: List[float] = field(default_factory=lambda: [0.6, 1.4])
    resolution_step:  float = 0.1
    min_size:         int   = 3          # communities smaller than this → tail
    backbone_k:       int   = 3          # draw only each node's k strongest links
    n_labels:         int   = 15         # name the n largest nodes on the graph
    # What node size shows — marker *area* proportional to it, so twice the
    # value reads twice as big: "citations" | "papers" | "fractional".
    size_by:          str   = "citations"
    size_range:       List[float] = field(default_factory=lambda: [5.0, 45.0])  # [min, max] diameter, px
    # What edge width shows — width proportional to its square root:
    # "count" (raw co-signed papers: volume) | "weight" (the normalised link,
    # as used for layout and communities: relative intensity).
    edge_width_by:    str   = "count"
    edge_width_range: List[float] = field(default_factory=lambda: [0.5, 6.0])  # [min, max] width, px
    # Node opacity: 1 hides the edges under the nodes; below 1 they show
    # through (Plotly's own default for sized markers is 0.7).
    node_opacity:     float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.node_opacity <= 1:
            raise ValueError("actors.network.node_opacity must be in (0, 1], "
                             f"got {self.node_opacity!r}")
        if self.edge_width_by not in ("count", "weight"):
            raise ValueError("actors.network.edge_width_by must be 'count' or "
                             f"'weight', got {self.edge_width_by!r}")
        if (len(self.edge_width_range) != 2
                or not 0 < self.edge_width_range[0] < self.edge_width_range[1]):
            raise ValueError("actors.network.edge_width_range must be [min, max] "
                             "with 0 < min < max")
        if self.size_by not in ("citations", "papers", "fractional"):
            raise ValueError("actors.network.size_by must be 'citations', 'papers' "
                             f"or 'fractional', got {self.size_by!r}")
        if len(self.size_range) != 2 or not 0 < self.size_range[0] < self.size_range[1]:
            raise ValueError("actors.network.size_range must be [min, max] with "
                             "0 < min < max")
        if self.normalize not in ("salton", "association", "count"):
            raise ValueError("actors.network.normalize must be 'salton', "
                             f"'association' or 'count', got {self.normalize!r}")
        if self.min_papers < 1 or self.max_nodes < 2:
            raise ValueError("actors.network needs min_papers >= 1 and max_nodes >= 2")


@dataclass
class ActorsSectionConfig(ConfigField):
    """Actors section: who drives the literature (authors, institutions, countries).

    Read from the reviewed dataset's ``actors`` column (built at the ``bib``
    stage, whatever the source); renders whenever that column is present. Per
    level in *levels*: the *top_n* most involved actors (ranked by *sort_by*:
    ``papers`` — full counting —, ``fractional`` — each paper split among its
    actors — or ``citations``) and, unless disabled, their collaboration network.
    """
    levels:  List[str] = field(default_factory=lambda: ["country", "institution", "author"])
    top_n:   int = 15
    sort_by: str = "papers"              # "papers" | "fractional" | "citations"
    network: ActorNetworkConfig = None

    def __post_init__(self):
        super().__post_init__()
        self.network = _with_defaults(ActorNetworkConfig, self.network, 'network')
        bad = [lv for lv in self.levels if lv not in ("author", "institution", "country")]
        if bad:
            raise ValueError(f"actors.levels: unknown level(s) {bad}; "
                             "expected author, institution, country")
        if self.sort_by not in ("papers", "fractional", "citations"):
            raise ValueError("actors.sort_by must be 'papers', 'fractional' or "
                             f"'citations', got {self.sort_by!r}")
        if self.top_n < 1:
            raise ValueError("actors.top_n must be >= 1")


@dataclass
class TemporalSectionConfig(ConfigField):
    # "trajectory" draws the per-year curves (counts and share of the year);
    # the other three are cumulative.
    variants: List[str] = field(default_factory=lambda: [
        "trajectory", "absolute", "normalized", "weighted"
    ])
    # Width, in years, of the centred rolling mean added beside the raw
    # trajectory curves. 1 = no smoothing (raw counts only).
    smoothing_window: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.smoothing_window < 1:
            raise ValueError(
                f"temporal.smoothing_window must be >= 1 (1 = no smoothing), "
                f"got {self.smoothing_window}"
            )


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
class ReviewSelectionConfig(ConfigField):
    """Tuning of the review branch of the reading list (reviews are added beside
    the `selection_by` ranking, then ranked and capped on their own)."""
    enabled:               bool = True
    # OpenAlex types reviews far more sparingly than WoS, so the type alone misses
    # the reviews an OpenAlex-heavy corpus only declares in its title.
    detect:                str  = "type_or_title"  # "type" | "title" | "type_or_title"
    max_reviews_per_topic: int  = 3        # historical cohort, best-first; no uncapped value
    relevance_mode:        str  = "blend"  # "blend" (cit/yr + raw citations) | "cpy" (cit/yr alone)
    # A review this recent has had no time to be cited, so it bypasses the
    # relevance ranking and keeps a guaranteed slot instead.
    recent_years:          int  = 1        # years back from the corpus's last year; 0 = no cohort
    max_recent_reviews:    int  = 2        # that cohort's own cap, newest first

    def __post_init__(self):
        super().__post_init__()
        if self.detect not in ("type", "title", "type_or_title"):
            raise ValueError(
                f"paper_selection.review.detect must be 'type', 'title' or "
                f"'type_or_title', got {self.detect!r}"
            )
        if self.relevance_mode not in ("blend", "cpy"):
            raise ValueError(
                f"paper_selection.review.relevance_mode must be 'blend' or "
                f"'cpy', got {self.relevance_mode!r}"
            )
        if self.max_reviews_per_topic < 1:
            raise ValueError(
                "paper_selection.review.max_reviews_per_topic must be >= 1; "
                "use enabled: false to drop the review branch entirely"
            )
        if self.recent_years < 0:
            raise ValueError(
                "paper_selection.review.recent_years must be >= 0 "
                "(0 = no guaranteed cohort for recent reviews)"
            )
        if self.max_recent_reviews < 1:
            raise ValueError(
                "paper_selection.review.max_recent_reviews must be >= 1; "
                "use recent_years: 0 to drop the recent cohort entirely"
            )


@dataclass
class PaperSelectionConfig(ConfigField):
    min_year:             int   = 2000
    # Global reading budget, shared out over the topics by eligible pool size —
    # the papers left after `min_year` and the research-front split, not the
    # topic's whole corpus. Reviews spend it too, so it is the number of papers
    # actually read. Unset = the legacy per-topic proportion below.
    budget:               Union[None, int] = None
    allocation:           str   = "sqrt"   # "sqrt" (compressed) | "linear" (strict)
    min_per_topic:        int   = 2        # floor, honoured even past `budget`
    max_per_topic:        Union[None, int] = None   # ceiling; unset = the pool
    # Used only when `budget` is unset: share of the topic's own corpus.
    proportion_per_topic: float = 0.15
    # "citations" | "random" | "coupling" | "co_citation" | "composite"
    # "composite" ranks papers per topic by the three-axis indicator (coupling
    # centrality + citation relevance + thematic representativeness); tune it
    # under the `composite:` sub-block.
    selection_by:         str   = "citations"
    composite:            CompositeConfig = None
    # Reviews are added independently of `selection_by`; tune them under `review:`.
    review:               ReviewSelectionConfig = None
    export_annex:         bool  = True
    annex_format:         str   = "csv"   # "csv" | "txt"

    def __post_init__(self):
        super().__post_init__()
        self.composite = _with_defaults(CompositeConfig, self.composite, 'composite')
        self.review    = _with_defaults(ReviewSelectionConfig, self.review, 'review')

        if self.allocation not in ("sqrt", "linear"):
            raise ValueError(
                f"paper_selection.allocation must be 'sqrt' or 'linear', "
                f"got {self.allocation!r}"
            )
        if self.budget is not None and self.budget < 1:
            raise ValueError("paper_selection.budget must be >= 1 (or unset)")
        if self.min_per_topic < 0:
            raise ValueError("paper_selection.min_per_topic must be >= 0")
        if self.max_per_topic is not None and self.max_per_topic < self.min_per_topic:
            raise ValueError(
                "paper_selection.max_per_topic must be >= min_per_topic"
            )


@dataclass
class ReportSectionsConfig(ConfigField):
    topics:                TopicsSectionConfig       = None
    bib_network:           BibNetworkSectionConfig   = None
    actors:                ActorsSectionConfig       = None
    temporal:              TemporalSectionConfig     = None
    topic_characteristics: TopicCharacteristicsConfig = None
    topic_similarity:      TopicSimilarityConfig     = None
    paper_selection:       PaperSelectionConfig      = None
    extra:                 Union[None, List[dict]]   = None

    def __post_init__(self):
        super().__post_init__()
        self.topics      = _with_defaults(TopicsSectionConfig, self.topics, 'topics')
        self.bib_network = _with_defaults(BibNetworkSectionConfig, self.bib_network, 'bib_network')
        self.actors      = _with_defaults(ActorsSectionConfig, self.actors, 'actors')
        self.temporal   = _with_defaults(TemporalSectionConfig, self.temporal, 'temporal')
        self.topic_characteristics = _with_defaults(
            TopicCharacteristicsConfig, self.topic_characteristics, 'topic_characteristics')
        self.topic_similarity = _with_defaults(
            TopicSimilarityConfig, self.topic_similarity, 'topic_similarity')
        self.paper_selection = _with_defaults(
            PaperSelectionConfig, self.paper_selection, 'paper_selection')


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
    requests_per_minute:     Union[None, int] = None   # None → provider default (Albert: 10)
    timeout:                 Union[None, float] = None # seconds per call; None → provider default
    n_repr_docs_for_labeling: int             = 3
    system_prompt:           Union[None, str] = None


@dataclass
class TopicReportExportConfig(ConfigField):
    """Output configuration for the topic-report stage.

    Each report is written to ``<export_dir>/report_<timestamp>/``.
    """
    export_dir: str


@dataclass
class TopicReportConfig(ConfigField):
    """PDF layout parameters for the topic-report stage."""
    run_dir:     str = None  # auto-detected by Config.load() from topic_model.export.export_dir when blank
    export:      Union[None, TopicReportExportConfig] = None
    meta:        Union[None, ReportMetaConfig]     = None
    sections:    Union[None, ReportSectionsConfig] = None

    def __post_init__(self):
        super().__post_init__()
        self.export   = _build(TopicReportExportConfig, self.export, 'export')
        self.meta     = _with_defaults(ReportMetaConfig, self.meta, 'meta')
        self.sections = _with_defaults(ReportSectionsConfig, self.sections, 'sections')


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
        self.unpaywall = _build(UnpaywallConfig, self.unpaywall, 'unpaywall')
        self.elsevier  = _build(ElsevierDownloadConfig, self.elsevier, 'elsevier')
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
        return _instantiate(cls, resolved)


#: Which config sections each stage reads. A ``needs`` section must be present
#: and valid for the stage to run; a ``uses`` one only decorates the stage, so a
#: broken one costs its contribution, not the run.
STAGE_SECTIONS = {
    'bib':          {'needs': ('bib',),          'uses': ()},
    'review':       {'needs': ('review',),       'uses': ('topic_model',)},
    'topic-model':  {'needs': ('topic_model',),  'uses': ('llm', 'topic_report')},
    'topic-report': {'needs': ('topic_report',), 'uses': ('topic_model', 'llm')},
}

#: Section name -> the dataclass mirroring it.
SECTION_CLASSES = {}   # filled in below, once every class exists


def mandatory_keys(cls, prefix: str = '') -> List[str]:
    """The dotted keys the YAML must fill under *cls*, sub-blocks included."""
    keys = []
    for field_ in fields(cls):
        if not _is_mandatory(field_):
            continue
        path = f'{prefix}.{field_.name}' if prefix else field_.name
        nested = _config_class(field_.type)
        if nested is None:
            keys.append(path)
        else:
            suffix = '[]' if get_origin(field_.type) is list else ''
            keys.extend(mandatory_keys(nested, path + suffix))
    return keys


def conditional_keys(cls, prefix: str = '') -> List[str]:
    """Keys mandatory only under some other setting, as declared by the
    dataclasses' ``CONDITIONAL`` notes."""
    notes = [f'{prefix} — {note}' for note in getattr(cls, 'CONDITIONAL', ())]
    for field_ in fields(cls):
        nested = _config_class(field_.type)
        if nested is not None and nested is not cls:
            path = f'{prefix}.{field_.name}' if prefix else field_.name
            notes.extend(conditional_keys(nested, path))
    return notes


def _config_class(annotation):
    """The ConfigField subclass an annotation resolves to, if any —
    unwrapping ``Union[None, X]`` and ``List[X]``."""
    if isinstance(annotation, type):
        return annotation if issubclass(annotation, ConfigField) else None
    for arg in get_args(annotation):
        found = _config_class(arg)
        if found is not None:
            return found
    return None


@dataclass
class Config:
    """Root configuration object.

    All sections are optional — only the sections present in the YAML are
    executed. The canonical stage order is:
    ``bib → review → topic-model → topic-report``.

    ``Config.load()`` propagates outputs between stages automatically when
    ``doc_dataset`` / ``run_dir`` are left blank, so a full-pipeline YAML
    requires no explicit cross-section paths.

    A section the YAML declares but gets wrong lands in :attr:`errors` with its
    attribute left None, so only the stages that need it fail (see
    :meth:`require_stages`).
    """
    env:                Union[None, str]                    = None
    #: Project root: each section exports to ``<output_dir>/<section>/``
    #: unless its own ``export.export_dir`` says otherwise.
    output_dir:         Union[None, str]                    = None
    bib:                Union[None, BibConfig]              = None
    review:             Union[None, ReviewConfig]           = None
    topic_model:        Union[None, TopicModelConfig]       = None
    topic_report:       Union[None, TopicReportConfig]      = None
    llm:                Union[None, TopicLabelerConfig]     = None
    #: Sections written in the YAML, whether or not they parsed.
    declared:           Set[str]                            = field(default_factory=set)
    #: Section name -> why it could not be parsed.
    errors:             Dict[str, ConfigError]              = field(default_factory=dict)

    # ---- per-stage validation ------------------------------------------

    def stage_is_declared(self, stage: str) -> bool:
        """True when the YAML declares every section *stage* needs."""
        return all(s in self.declared for s in STAGE_SECTIONS[stage]['needs'])

    def require_stages(self, stages: List[str]) -> None:
        """Raise on the first section *stages* need and cannot get; warn for the
        ones they merely use."""
        for stage in stages:
            for name in STAGE_SECTIONS[stage]['needs']:
                if name in self.errors:
                    raise self.errors[name]
                if getattr(self, name) is None:
                    raise ConfigError(
                        f"stage {stage!r} needs a `{name}:` section in the config. "
                        f"Mandatory keys: "
                        f"{', '.join(mandatory_keys(SECTION_CLASSES[name], name)) or 'none'}.",
                        path=name)
            for name in STAGE_SECTIONS[stage]['uses']:
                if name in self.errors:
                    print(f"[config] stage {stage!r} runs without `{name}:` — "
                          f"{self.errors[name]}")

    @classmethod
    def load(cls, config_file):
        """Load a YAML config file.

        Steps:
          1. Read the YAML.
          2. Load the .env file referenced by the root-level ``env:`` key (if any).
          3. Resolve all ``${VAR}`` references.
          4. Place each section's ``export_dir`` under the root ``output_dir``.
          5. Propagate outputs between stages when ``doc_dataset`` / ``run_dir``
             are left blank (auto-detection of the latest run in each export_dir).
          6. Build a typed dataclass for every section present, recording rather
             than raising the failures (see :meth:`require_stages`).
        """
        with open(config_file, 'r') as file:
            raw = yaml.safe_load(file) or {}

        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        resolved = _resolve_export_dirs(_resolve_env_vars(raw))
        output_dir = resolved.get('output_dir')

        # ── Propagate outputs between stages when doc_dataset is blank ────────
        #   bib export dir    → review.doc_dataset
        #   review export dir → topic_model.doc_dataset
        #     (topic_model.doc_dataset — the reviewed_included.csv — is also the
        #      source the report's network panels are recomputed from.)
        bib_export_dir    = _export_dir_of(resolved, 'bib')
        review_export_dir = _export_dir_of(resolved, 'review')
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
            tm_export_dir = _export_dir_of(resolved, 'topic_model')
            if tm_export_dir:
                latest = _find_latest_dir(tm_export_dir)
                if latest:
                    topic_report_data['run_dir'] = latest

        # A section name the loader does not know would otherwise drop a whole
        # stage without a word — a typo must not read as "not configured".
        for name in resolved:
            if name not in ('env', 'output_dir') and name not in SECTION_CLASSES:
                print(f"[config] ignoring unknown section `{name}:` — known "
                      f"sections: {', '.join(SECTION_CLASSES)}")

        declared, errors, sections = set(), {}, {}
        for name, data in (('bib',          resolved.get('bib')),
                           ('review',       review_data),
                           ('topic_model',  topic_model_data),
                           ('topic_report', topic_report_data),
                           ('llm',          resolved.get('llm'))):
            if not data:
                continue
            declared.add(name)
            try:
                with _at(name):
                    sections[name] = _instantiate(SECTION_CLASSES[name], data)
            except (ValueError, TypeError) as exc:
                # A section is only wrong for the stages that read it: record it,
                # let `require_stages` raise it if and when one of them runs.
                if not output_dir and re.search(r'missing.*\bexport\b', str(exc)):
                    exc = ConfigError(
                        f"{exc} — or set a root-level `output_dir:` to export "
                        f"every section under one project folder.",
                        path=getattr(exc, 'path', name))
                errors[name] = exc if isinstance(exc, ConfigError) \
                    else ConfigError(str(exc), path=name)
                print(f"[config] section `{name}:` is unusable and will stop the "
                      f"stages that need it — {exc}")

        return cls(env=resolved.get('env'), output_dir=output_dir,
                   declared=declared, errors=errors, **sections)


SECTION_CLASSES.update(
    bib          = BibConfig,
    review       = ReviewConfig,
    topic_model  = TopicModelConfig,
    topic_report = TopicReportConfig,
    llm          = TopicLabelerConfig,
)
