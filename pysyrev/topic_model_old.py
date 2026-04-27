from dataclasses import dataclass
from typing import List

from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from pysyrev.core.topic import clean_dataset, topic_modeling


@dataclass
class UmapModel:

    min_dist : float
    metric : str
    low_memory : bool
    random_state : int

    def __call__(self, n_neighbors, n_components, embeddings):

        return UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=self.min_dist,
            metric=self.metric,
            low_memory=self.low_memory,
            random_state=self.random_state
        ).fit_transform(embeddings)


@dataclass
class HdbscanModel:

    metric : str
    cluster_selection_method : str
    prediction_data : bool

    def __call__(self, min_topic_size, min_samples):

        return HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=self.prediction_data
        )


@dataclass
class BertopicModel:

    hdbscan_model : HdbscanModel
    umap_model : UmapModel
    ctfidf_model : ClassTfidfTransformer
    embedding_model : SentenceTransformer
    calculate_probabilities : bool
    n_gram_range : str
    language : str
    nr_topics = None
    verbose : bool = False

    def __call__(self, min_topic_size, min_samples):

        return dict(
            embedding_model=self.embedding_model,
            nr_topics=self.nr_topics,
            n_gram_range=self.n_gram_range,
            verbose=self.verbose,
            language=self.language,
            calculate_probabilities=self.calculate_probabilities,
            ctfidf_model=self.ctfidf_model,
            umap_model=BaseDimensionalityReduction(),
            hdbscan_model=self.hdbscan_model(min_topic_size, min_samples)
        )


@dataclass
class TopicDist:

    window : int
    stride : int
    min_similarity : float
    batch_size : int


@dataclass
class TopicModel:

    allow_abbrev : bool
    distance_name : str
    bertopic_model : BertopicModel
    topic_distribution : TopicDist
    nr_repr_docs : int
    export_to : str
    n_neighbors : List[int]
    n_components : List[int]
    min_topic_size : List[int]
    min_samples : List[int]
    topic_size_step : int
    min_sample_step : int
    keep_n_results : int


    def _clean_dataset(self, dataset, show_progress):

        return clean_dataset(dataset, self.allow_abbrev, show_progress)

    def run(self,
            dataset,
            show_progress=True):

        cleans_docs = self._clean_dataset(dataset,
                                          show_progress=show_progress)
        embeddings = self.bertopic_model.embedding_model.encode(cleans_docs,
                                                                show_progress_bar=show_progress)

        return topic_modeling(dataset,
                              cleans_docs,
                              self.bertopic_model,
                              self.topic_distribution,
                              embeddings,
                              self.n_neighbors,
                              self.n_components,
                              self.min_topic_size,
                              self.min_samples,
                              self.topic_size_step,
                              self.min_sample_step,
                              self.export_to,
                              self.nr_repr_docs,
                              self.distance_name,
                              self.keep_n_results,
                              show_progress)

