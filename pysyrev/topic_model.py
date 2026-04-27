"""
Runtime model classes for the topic-modeling pipeline.

These classes hold *configured* runtime objects (a UMAP factory, an HDBSCAN
factory, a SentenceTransformer instance...) and are instantiated either
directly or from a parsed config via TopicModel.from_config().

Field names mirror the corresponding entries in config.py to keep the
bridge between the two trivial.
"""

from dataclasses import dataclass
from typing import List

from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from pysyrev.config import Config, TopicModelConfig
from pysyrev.core.topic import clean_dataset, topic_modeling


# =============================================================================
# Runtime sub-models
# =============================================================================

@dataclass
class UmapModel:
    min_dist:     float
    metric:       str
    low_memory:   bool
    random_state: int

    def __call__(self, n_neighbors, n_components, embeddings):
        return UMAP(
            n_neighbors  = n_neighbors,
            n_components = n_components,
            min_dist     = self.min_dist,
            metric       = self.metric,
            low_memory   = self.low_memory,
            random_state = self.random_state,
        ).fit_transform(embeddings)


@dataclass
class HdbscanModel:
    metric:                   str
    cluster_selection_method: str
    prediction_data:          bool

    def __call__(self, min_topic_size, min_samples):
        return HDBSCAN(
            min_cluster_size         = min_topic_size,
            min_samples              = min_samples,
            metric                   = self.metric,
            cluster_selection_method = self.cluster_selection_method,
            prediction_data          = self.prediction_data,
        )


@dataclass
class BertopicModel:
    hdbscan_model:           HdbscanModel
    umap_model:              UmapModel
    ctfidf_model:            ClassTfidfTransformer
    transformer_model:       str
    calculate_probabilities: bool
    n_gram_range:            str
    language:                str
    nr_topics:               int = None
    verbose:                 bool = False

    # Cache for the loaded SentenceTransformer (filled on first access).
    _embedding_model:        SentenceTransformer = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazily load the SentenceTransformer on first access. Subsequent
        accesses reuse the same instance, so multiple `run()` calls on the
        same TopicModel share the loaded weights."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.transformer_model)
        return self._embedding_model

def __call__(self, min_topic_size, min_samples):
        return dict(
            embedding_model         = self.embedding_model,
            nr_topics               = self.nr_topics,
            n_gram_range            = self.n_gram_range,
            verbose                 = self.verbose,
            language                = self.language,
            calculate_probabilities = self.calculate_probabilities,
            ctfidf_model            = self.ctfidf_model,
            umap_model              = BaseDimensionalityReduction(),
            hdbscan_model           = self.hdbscan_model(min_topic_size, min_samples),
        )


@dataclass
class TopicDistribution:
    window:         int
    stride:         int
    min_similarity: float
    batch_size:     int


# =============================================================================
# Top-level runtime model
# =============================================================================

@dataclass
class TopicModel:
    allow_abbrev:         bool
    distance_name:        str
    bertopic_model:       BertopicModel
    topic_distribution:   TopicDistribution
    nr_repr_docs:         int
    export_to:            str
    n_neighbors:          List[int]
    n_components:         List[int]
    min_topic_size_range: List[int]
    min_sample_range:     List[int]
    topic_size_step:      int
    min_sample_step:      int
    keep_n_results:       int

    # ---- bridge from configuration --------------------------------------

    @classmethod
    def from_config(cls,
                    config: TopicModelConfig) -> 'TopicModel':
        """
        Build a TopicModel from a parsed TopicModelConfig.

        """
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting        = config.ctfidf.bm25_weighting,
            reduce_frequent_words = config.ctfidf.reduce_frequent_words,
        )
        umap_model = UmapModel(
            min_dist     = config.umap.min_dist,
            metric       = config.umap.metric,
            low_memory   = config.umap.low_memory,
            random_state = config.umap.random_state,
        )
        hdbscan_model = HdbscanModel(
            metric                   = config.hdbscan.metric,
            cluster_selection_method = config.hdbscan.cluster_selection_method,
            prediction_data          = config.hdbscan.prediction_data,
        )
        bertopic_model = BertopicModel(
            hdbscan_model           = hdbscan_model,
            umap_model              = umap_model,
            ctfidf_model            = ctfidf_model,
            transformer_model       = config.bertopic.transformer_model,
            calculate_probabilities = config.bertopic.calculate_probabilities,
            n_gram_range            = config.bertopic.n_gram_range,
            language                = config.bertopic.language,
        )
        topic_distribution = TopicDistribution(
            window         = config.topic_distribution.window,
            stride         = config.topic_distribution.stride,
            min_similarity = config.topic_distribution.min_similarity,
            batch_size     = config.topic_distribution.batch_size,
        )
        return cls(
            allow_abbrev         = config.berteley.allow_abbrev,
            distance_name        = config.distance_name,
            bertopic_model       = bertopic_model,
            topic_distribution   = topic_distribution,
            nr_repr_docs         = config.bertopic.nr_repr_docs,
            export_to            = config.export_to,
            n_neighbors          = config.umap.n_neighbors,
            n_components         = config.umap.n_components,
            min_topic_size_range = config.hdbscan.min_topic_size_range,
            min_sample_range     = config.hdbscan.min_sample_range,
            topic_size_step      = config.hdbscan.topic_size_step,
            min_sample_step      = config.hdbscan.min_sample_step,
            keep_n_results       = config.keep_n_results,
        )

    # ---- runtime --------------------------------------------------------

    def _clean_dataset(self, dataset, show_progress):
        return clean_dataset(dataset, self.allow_abbrev, show_progress)

    def run(self, dataset, show_progress=True):
        cleans_docs = self._clean_dataset(dataset, show_progress=show_progress)
        embeddings = self.bertopic_model.embedding_model.encode(
            cleans_docs, show_progress_bar=show_progress,
        )
        return topic_modeling(
            dataset,
            cleans_docs,
            self.bertopic_model,
            self.topic_distribution,
            embeddings,
            self.n_neighbors,
            self.n_components,
            self.min_topic_size_range,
            self.min_sample_range,
            self.topic_size_step,
            self.min_sample_step,
            self.export_to,
            self.nr_repr_docs,
            self.distance_name,
            self.keep_n_results,
            show_progress,
        )
