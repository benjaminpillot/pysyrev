"""
Configuration loader.

Reads a YAML file and produces typed configuration dataclasses. These
classes know nothing about the business model — they only mirror the YAML
structure. The bridge from config to the runtime model lives in
topic_model.py via TopicModel.from_config().

Naming convention: every config dataclass ends with `Config` so it is
unambiguous when imported alongside the runtime classes (e.g. UMAPConfig
vs UmapModel).
"""

import dataclasses
from dataclasses import dataclass, fields
from typing import List, Union

import yaml


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
class BibFileConfig(ConfigField):
    wos:        str
    open_alex:  str
    scopus:     str
    pubmed:     str
    doc_type:   List[str]


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
    env:                str
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
    distance_name:       str
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
    bibfile:     BibFileConfig
    review:      ReviewConfig
    topic_model: TopicModelConfig

    @classmethod
    def load(cls, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return cls(
            bibfile     = BibFileConfig(**config['bibfile']),
            review      = ReviewConfig(**config['review']),
            topic_model = TopicModelConfig(**config['topic_model']),
        )
