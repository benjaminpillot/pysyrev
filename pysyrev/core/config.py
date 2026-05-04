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
import os
import re
from dataclasses import dataclass, fields
from typing import List, Union

import yaml
from dotenv import load_dotenv


# Pattern to resolve ${ENV_VAR} references inside YAML string values.
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _resolve_env_vars(node):
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
        for field in fields(self):
            if (not isinstance(field.default, dataclasses._MISSING_TYPE)
                    and getattr(self, field.name) is None):
                setattr(self, field.name, field.default)


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
    doc_type:     Union[None, List[str]]             = None
    year:         int                                = 1900
    nb_citations: int                                = 0
    language:     Union[None, str, List[str]]        = None
    scorer:       str                                = "partial_token_sort_ratio"
    score_cutoff: int                                = 90


@dataclass
class ResolveReferencesConfig(ConfigField):
    enabled:            bool = False
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
class BibConfig(ConfigField):
    wos:               Union[None, str, WosSourceConfig]
    open_alex:         Union[None, str, OpenAlexSourceConfig]
    scopus:            Union[None, str]
    pubmed:            Union[None, str]
    export_to:          Union[None, str]                           = None
    clean:             Union[None, dict, CleanConfig]             = None
    extract:           Union[None, dict, ExtractConfig]           = None
    resolve_references: Union[None, dict, ResolveReferencesConfig] = None
    merge:             Union[None, dict, MergeConfig]             = None

    def __post_init__(self):
        super().__post_init__()
        # Backwards compatibility: a string under `wos:` / `open_alex:` is
        # treated as a file path. A dict is parsed as a structured source config.
        if isinstance(self.wos, dict):
            self.wos = WosSourceConfig(**self.wos)
        elif isinstance(self.wos, str):
            self.wos = WosSourceConfig(source='file', file=self.wos)

        if isinstance(self.open_alex, dict):
            self.open_alex = OpenAlexSourceConfig(**self.open_alex)
        elif isinstance(self.open_alex, str):
            self.open_alex = OpenAlexSourceConfig(source='file', file=self.open_alex)

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
    min_topic_size_range:     List[int]
    min_sample_range:         List[int]
    topic_size_step:          int = 1
    min_sample_step:          int = 1
    cluster_selection_method: str = 'leaf'
    metric:                   str = 'euclidean'
    prediction_data:          bool = True


@dataclass
class UMAPConfig(ConfigField):
    n_neighbors:  List[int]
    n_components: List[int]
    metric:       str   = 'cosine'
    min_dist:     float = 0.0
    low_memory:   bool  = False
    random_state: int   = 42


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
class ReviewerConfig(ConfigField):
    """Mirror of one entry under `review.reviewers` in the YAML.
    Cross-section fields like inclusion_criteria are NOT here — they live
    at the ReviewConfig level and are wired together by the runtime layer."""
    model_id:           str
    host:               str
    provider:           str
    name:               str
    max_tokens:         int
    temperature:        float
    reasoning_effort:   str
    backstory:          str
    additional_context: str
    reasoning:          str = 'brief'


@dataclass
class ReviewConfig(ConfigField):
    export_to:          str
    batch_size:         int
    api_pause:          float
    text_inputs:        List[str]
    inclusion_criteria: str
    exclusion_criteria: str
    decision_rule:      str
    reviewers:          List[ReviewerConfig]
    sampling:           bool
    sample_size:        Union[None, int]
    workflow:           List[dict]
    run_name:           Union[None, str] = None  # None → default filenames

    def __post_init__(self):
        super().__post_init__()
        # Mirror nested YAML dicts as ReviewerConfig instances.
        self.reviewers = [ReviewerConfig(**r) for r in self.reviewers]
        if not self.sampling:
            self.sample_size = None


@dataclass
class TopicModelConfig(ConfigField):
    doc_dataset:         str
    export_to:           str
    distance:            str
    keep_n_results:      int
    coherence_scorer:    CoherenceScorerConfig
    hdbscan:             HDBSCANConfig
    umap:                UMAPConfig
    bertopic:            BertopicConfig
    berteley:            BerteleyConfig
    ctfidf:              CTFIDFConfig
    topic_distribution:  TopicDistributionConfig
    run_name:            Union[None, str] = None  # None -> auto-timestamp at run time

    def __post_init__(self):
        super().__post_init__()
        self.hdbscan            = HDBSCANConfig(**self.hdbscan)
        self.umap               = UMAPConfig(**self.umap)
        self.bertopic           = BertopicConfig(**self.bertopic)
        self.berteley           = BerteleyConfig(**self.berteley)
        self.ctfidf             = CTFIDFConfig(**self.ctfidf)
        self.coherence_scorer   = CoherenceScorerConfig(**self.coherence_scorer)
        self.topic_distribution = TopicDistributionConfig(**self.topic_distribution)


@dataclass
class Config:
    """Root configuration object."""
    bib:         BibConfig
    review:      ReviewConfig
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
          4. Build typed dataclasses.
        """
        with open(config_file, 'r') as file:
            raw = yaml.safe_load(file)

        # Step 2: load .env BEFORE resolving ${VAR}, so variables it defines
        # are available for resolution.
        env_path = raw.get('env')
        if env_path:
            load_dotenv(env_path)

        # Step 3: walk the structure and substitute ${VAR}.
        resolved = _resolve_env_vars(raw)

        # Step 4: build typed config.
        return cls(
            env         = resolved.get('env'),
            bib         = BibConfig(**resolved['bib']),
            review      = ReviewConfig(**resolved['review']),
            topic_model = TopicModelConfig(**resolved['topic_model']),
        )
