import dataclasses
from dataclasses import dataclass, fields
from typing import List, Union

import yaml


@dataclass
class ConfigField(dict):

    def __post_init__(self):

        for field in fields(self):
            if (not isinstance(field.default, dataclasses._MISSING_TYPE)
                    and getattr(self, field.name) is None):
                setattr(self, field.name, field.default)



@dataclass
class BibFile(ConfigField):

    wos : str
    open_alex : str
    scopus : str
    pubmed : str


@dataclass
class HDBSCAN(ConfigField):

    min_topic_size_range : List[int]
    min_sample_range : List[int]
    topic_size_step : int = 1
    min_sample_step : int = 1
    cluster_selection_method : str = "leaf"
    metric : str = "euclidean"


@dataclass
class UMAP(ConfigField):

    n_neighbors : List[int]
    n_components : List[int]
    metric : str = "cosine"
    min_dist : float = 0.0
    low_memory : bool = False
    random_state : int = 42


@dataclass
class Berteley(ConfigField):

    allow_abbrev : bool = False


@dataclass
class CTFIDF(ConfigField):

    bm25_weighting : bool = True
    reduce_frequent_words : bool = True


@dataclass
class TopicDistribution(ConfigField):

    window : int = 8
    stride : int = 1
    similarity : float = 0.1
    batch_size : int = 1000


@dataclass
class Bertopic(ConfigField):

    transformer_model : str = "allenai/specter2_base"
    n_gram_range : str = "bigram"
    language : str = "english"
    nr_repr_docs : int = 50
    calculate_probabilities : bool = True


@dataclass
class Reviewer(ConfigField):

    model_id : str
    host : str
    inclusion_criteria : str
    exclusion_criteria : str
    input_description : str
    provider : str
    name : str
    max_tokens : int
    temperature : float
    reasoning_effort : str
    backstory : str
    additional_context : str
    reasoning : str = "brief"


@dataclass
class Review(ConfigField):

    env : str
    export_to : str
    batch_size : int
    api_pause : float
    text_inputs : List[str]
    inclusion_criteria : str
    exclusion_criteria : str
    decision_rule : str
    reviewers : List[dict]
    sampling : bool
    sample_size : Union[None, int]
    workflow: List[dict]

    def __post_init__(self):
        self.input_description = (f"article {self.text_inputs[0]}/"
                                  f"{self.text_inputs[1]}/{self.text_inputs[2]}")
        self.reviewers = [Reviewer(**r,
                                   input_description=self.input_description,
                                   inclusion_criteria=self.inclusion_criteria,
                                   exclusion_criteria=self.exclusion_criteria).__dict__
                          for r in self.reviewers]

        if not self.sampling:
            self.sample_size = None

@dataclass
class TopicModeling(ConfigField):

    doc_type : List[str]
    doc_dataset : str
    export_to : str
    hdbscan : HDBSCAN
    umap : UMAP
    bertopic : Bertopic
    berteley : Berteley
    ctfidf : CTFIDF
    topic_distribution : TopicDistribution

    def __post_init__(self):

        self.hdbscan = HDBSCAN(**self.hdbscan)
        self.umap = UMAP(**self.umap)
        self.bertopic = Bertopic(**self.bertopic)
        self.berteley = Berteley(**self.berteley)
        self.ctfidf = CTFIDF(**self.ctfidf)
        self.topic_distribution = TopicDistribution(**self.topic_distribution)


@dataclass
class Config:

    bibfile : BibFile
    review : Review
    topic_modeling : TopicModeling


    @classmethod
    def load(cls, config_file):

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        return cls(BibFile(**config["bibfile"]),
                   Review(**config["review"]),
                   TopicModeling(**config["topic_modeling"]))
