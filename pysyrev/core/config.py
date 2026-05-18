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
from typing import List, Union

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
    """One OpenAlex source: either a file path, or an API config. Exactly
    one of `file` / `api` must be set."""
    source: str = 'file'  # 'file' or 'api'
    file:   Union[None, str] = None
    api:    Union[None, OpenAlexApiConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.source == 'api' and isinstance(self.api, dict):
            self.api = OpenAlexApiConfig(**self.api)
        if self.source == 'file' and not self.file:
            raise ValueError("OpenAlex source is 'file' but no `file:` path is set")
        if self.source == 'api' and self.api is None:
            raise ValueError("OpenAlex source is 'api' but no `api:` block is set")
        if self.source not in ('file', 'api'):
            raise ValueError(f"Unknown OpenAlex source {self.source!r}; expected 'file' or 'api'")


@dataclass
class CleanConfig(ConfigField):
    min_signals_to_reject: int                    = 2
    extra_garbage_phrases: Union[None, List[str]] = None
    use_langdetect:        bool                   = False


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


@dataclass
class HDBSCANConfig(ConfigField):
    min_topic_size_range:     List[int] = field(default_factory=lambda: [2, 2])
    min_sample_range:         List[int] = field(default_factory=lambda: [2, 2])
    topic_size_step:          int       = 1
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
    nr_repr_docs:            int  = 50
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
    max_retries:             Union[None, int] = None          # None → module default (2);  overridable per reviewer
    max_concurrent_requests: Union[None, int] = None          # None → module default (10); overridable per reviewer
    items_per_call:          Union[None, int] = None          # None → module default (1);  overridable per reviewer

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.export, dict):
            self.export = ReviewExportConfig(**self.export)
        self.reviewers = [ReviewerConfig(**r) for r in self.reviewers]


@dataclass
class CouplingNetworkConfig(ConfigField):
    use_resolved:   bool = False
    use_unresolved: bool = False
    min_shared:     int = 1


@dataclass
class CocitationNetworkConfig(ConfigField):
    use_resolved:    bool = False
    use_unresolved:  bool = False
    min_cocitations: int = 1


@dataclass
class BibNetworkExportConfig(ConfigField):
    """Output configuration for the bib_network stage.

    Each run is stored in ``<export_dir>/<run_name>/``.
    Leave ``run_name`` blank to auto-generate a timestamp.
    Call :meth:`resolve` to finalise the run directory and set file paths.
    """
    export_dir:       str
    run_name:         Union[None, str] = None
    coupling_graph:   Union[None, str] = None   # set by resolve()
    cocitation_graph: Union[None, str] = None   # set by resolve()

    def resolve(self):
        """Create the run directory and set output file paths."""
        self.run_name, run_dir = _make_run_dir(self.export_dir, self.run_name)
        self.coupling_graph   = os.path.join(run_dir, 'coupling_network.graphml')
        self.cocitation_graph = os.path.join(run_dir, 'cocitation_network.graphml')
        return self


@dataclass
class BibNetworkConfig(ConfigField):
    doc_dataset:        str                                 = None
    coupling_network:   CouplingNetworkConfig               = None
    cocitation_network: CocitationNetworkConfig             = None
    export:             Union[None, BibNetworkExportConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.coupling_network, dict):
            self.coupling_network = CouplingNetworkConfig(**self.coupling_network)
        elif self.coupling_network is None:
            self.coupling_network = CouplingNetworkConfig()

        if isinstance(self.cocitation_network, dict):
            self.cocitation_network = CocitationNetworkConfig(**self.cocitation_network)
        elif self.cocitation_network is None:
            self.cocitation_network = CocitationNetworkConfig()

        if isinstance(self.export, dict):
            self.export = BibNetworkExportConfig(**self.export)


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
    coherence_scorer:    CoherenceScorerConfig                  = None
    hdbscan:             HDBSCANConfig                          = None
    umap:                UMAPConfig                             = None
    bertopic:            BertopicConfig                         = None
    berteley:            BerteleyConfig                         = None
    ctfidf:              CTFIDFConfig                           = None
    topic_distribution:  TopicDistributionConfig                = None

    def __post_init__(self):
        super().__post_init__()

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
class ReportMetaConfig(ConfigField):
    title:       str              = "Bibliographic report — Pysyrev"
    subtitle:    Union[None, str] = None
    author:      str              = "Report generated with the pysyrev engine (v0.1)"
    date_format: str              = "%d/%m/%Y"
    version:     str              = "1.0.0"
    summary:     Union[None, str] = None


@dataclass
class ReportConfig(ConfigField):
    meta:     Union[None, ReportMetaConfig] = None
    sections: Union[None, List[dict]]       = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.meta, dict):
            self.meta = ReportMetaConfig(**self.meta)
        elif self.meta is None:
            self.meta = ReportMetaConfig()


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
    nr_repr_docs:            int              = 3
    system_prompt:           Union[None, str] = None


@dataclass
class TopicReportConfig(ConfigField):
    """Model-selection parameters for the topic-report stage."""
    run_dir:     str = None  # resolved by TopicReportFileConfig.load()
    model_index: int = 0
    export_to:   str = None


@dataclass
class BibNetworkReportConfig(ConfigField):
    """Paths to the exported bib_network graphs for inclusion in the report.

    Leave both paths blank and set 'config' at the root of the report YAML so
    that the latest bib_network run is detected automatically from
    bib_network.export.export_dir in the main pipeline config.
    """
    coupling_graph:   Union[None, str] = None
    cocitation_graph: Union[None, str] = None


@dataclass
class TopicReportFileConfig:
    """Root config for a report YAML file. Analogous to Config for the main pipeline.

    Five independent top-level sections:
      - config:       path to the main pysyrev config (used to auto-detect the latest
                      topic_model run when topic_report.run_dir is left blank, and the
                      latest bib_network run when bib_network paths are blank)
      - topic_report: which run and which model to use
      - bib_network:  optional — coupling / co-citation graph files to include
      - llm:          optional LLM labeler to generate human-readable topic names
      - report:       PDF layout (metadata, optional extra sections)
    """
    topic_report: TopicReportConfig
    report:       ReportConfig
    llm:          Union[None, TopicLabelerConfig]      = None
    bib_network:  Union[None, BibNetworkReportConfig]  = None
    env:          Union[None, str]                     = None
    config:       Union[None, str]                     = None

    @classmethod
    def load(cls, config_file):
        with open(config_file, 'r') as f:
            raw = yaml.safe_load(f)

        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        resolved = _resolve_env_vars(raw)

        main_raw = None

        def _load_main_raw():
            nonlocal main_raw
            if main_raw is not None:
                return main_raw
            main_config_path = resolved.get('config')
            if not main_config_path:
                return None
            with open(main_config_path, 'r') as f:
                main_raw = yaml.safe_load(f)
            return main_raw

        # ---- topic_report: auto-detect latest run when run_dir is blank ----
        tr_raw = dict(resolved.get('topic_report', {}))
        if not tr_raw.get('run_dir'):
            main = _load_main_raw()
            if not main:
                raise ValueError(
                    "topic_report.run_dir is blank: set 'config' (path to the main "
                    "pysyrev config) so that the latest topic_model run can be "
                    "detected automatically."
                )
            export_dir = (main.get('topic_model') or {}).get('export', {}).get('export_dir')
            if not export_dir:
                raise ValueError(
                    f"topic_model.export.export_dir is not set in the main config."
                )
            latest = _find_latest_dir(export_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"No topic_model run directories found in {export_dir!r}."
                )
            tr_raw['run_dir'] = latest

        # ---- bib_network: auto-detect graph paths when blank ---------------
        bn_raw = dict(resolved.get('bib_network') or {})
        if not bn_raw.get('coupling_graph') or not bn_raw.get('cocitation_graph'):
            main = _load_main_raw()
            if main:
                bn_export_dir = (
                    (main.get('bib_network') or {})
                    .get('export', {})
                    .get('export_dir')
                )
                if bn_export_dir:
                    latest_dir = _find_latest_dir(bn_export_dir)
                    if latest_dir:
                        if not bn_raw.get('coupling_graph'):
                            candidate = os.path.join(latest_dir, 'coupling_network.graphml')
                            if os.path.isfile(candidate):
                                bn_raw['coupling_graph'] = candidate
                        if not bn_raw.get('cocitation_graph'):
                            candidate = os.path.join(latest_dir, 'cocitation_network.graphml')
                            if os.path.isfile(candidate):
                                bn_raw['cocitation_graph'] = candidate

        llm_raw = resolved.get('llm')
        return cls(
            env          = resolved.get('env'),
            config       = resolved.get('config'),
            topic_report = TopicReportConfig(**tr_raw),
            report       = ReportConfig(**resolved.get('report', {})),
            llm          = TopicLabelerConfig(**llm_raw) if llm_raw else None,
            bib_network  = BibNetworkReportConfig(**bn_raw) if bn_raw else None,
        )


@dataclass
class Config:
    """Root configuration object."""
    bib:         BibConfig
    review:      ReviewConfig
    bib_network: BibNetworkConfig
    topic_model: TopicModelConfig
    env:         Union[None, str] = None  # path to .env file (loaded before ${VAR} resolution)

    @classmethod
    def load(cls, config_file):
        """
        Load a YAML config file. Steps:
          1. Read the YAML as raw dict.
          2. If `env:` (root-level) points to a .env file, load it so its
             variables become available in os.environ.
          3. Resolve all ${VAR} references throughout the structure.
          4. Auto-fill blank doc_dataset fields from the latest review run.
          5. Build typed dataclasses.
        """
        with open(config_file, 'r') as file:
            raw = yaml.safe_load(file)

        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        resolved = _resolve_env_vars(raw)

        # Step 4: propagate the latest output of each stage to the next one
        # when doc_dataset is left blank (standalone re-run use case).
        #   bib.export.export_dir      → review.doc_dataset
        #   review.export.export_dir   → bib_network.doc_dataset
        #                              → topic_model.doc_dataset
        bib_export_dir    = (resolved.get('bib') or {}).get('export', {}).get('export_dir')
        review_export_dir = (resolved.get('review') or {}).get('export', {}).get('export_dir')
        review_data       = resolved.get('review', {})
        bib_network_data  = resolved.get('bib_network', {})
        topic_model_data  = resolved.get('topic_model', {})

        if bib_export_dir and not review_data.get('doc_dataset'):
            latest = _find_latest_file(bib_export_dir, 'bib_dataset.csv')
            if latest:
                review_data['doc_dataset'] = latest

        if review_export_dir:
            latest = _find_latest_file(review_export_dir, 'reviewed_included.csv')
            if latest:
                if not bib_network_data.get('doc_dataset'):
                    bib_network_data['doc_dataset'] = latest
                if not topic_model_data.get('doc_dataset'):
                    topic_model_data['doc_dataset'] = latest

        return cls(
            env         = resolved.get('env'),
            bib         = BibConfig(**resolved['bib']),
            review      = ReviewConfig(**review_data),
            bib_network = BibNetworkConfig(**bib_network_data),
            topic_model = TopicModelConfig(**topic_model_data),
        )
