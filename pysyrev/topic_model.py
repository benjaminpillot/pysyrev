"""
Runtime model classes for the topic-modeling pipeline.

These classes hold *configured* runtime objects (a UMAP factory, an HDBSCAN
factory, a SentenceTransformer instance...) and are instantiated either
directly or from a parsed config via TopicModel.from_config().

Field names mirror the corresponding entries in config.py to keep the
bridge between the two trivial.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from pysyrev.core.config import TopicModelConfig
from pysyrev.core.topic import clean_dataset, topic_modeling
from pysyrev.core.topic_labels import load_cached_labels, save_labels


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

    doc_dataset:          str
    allow_abbrev:         bool
    distance:             str
    bertopic_model:       BertopicModel
    topic_distribution:   TopicDistribution
    nr_repr_docs:         int
    export_dir:           str
    n_neighbors:          List[int]
    n_components:         List[int]
    min_topic_size_range: List[int]
    min_sample_range:     List[int]
    topic_size_step:      int
    min_sample_step:      int
    keep_n_results:       int
    ranking_scorer:       str
    purity_scorer:        str
    run_name:             Union[None, str]
    labeler_config:       Optional[object] = None  # TopicLabelerConfig or None
    best_model_index:     int = 0
    overwrite:            bool = False
    _run_dir:             Union[None, str] = field(default=None, init=False, repr=False)

    # ---- bridge from configuration --------------------------------------

    @classmethod
    def from_config(cls, config: 'Config') -> 'TopicModel':
        """Build a TopicModel from a full Config object."""
        from pysyrev.core.config import Config
        tc = config.topic_model
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting        = tc.ctfidf.bm25_weighting,
            reduce_frequent_words = tc.ctfidf.reduce_frequent_words,
        )
        umap_model = UmapModel(
            min_dist     = tc.umap.min_dist,
            metric       = tc.umap.metric,
            low_memory   = tc.umap.low_memory,
            random_state = tc.umap.random_state,
        )
        hdbscan_model = HdbscanModel(
            metric                   = tc.hdbscan.metric,
            cluster_selection_method = tc.hdbscan.cluster_selection_method,
            prediction_data          = tc.hdbscan.prediction_data,
        )
        bertopic_model = BertopicModel(
            hdbscan_model           = hdbscan_model,
            umap_model              = umap_model,
            ctfidf_model            = ctfidf_model,
            transformer_model       = tc.bertopic.transformer_model,
            calculate_probabilities = tc.bertopic.calculate_probabilities,
            n_gram_range            = tc.bertopic.n_gram_range,
            language                = tc.bertopic.language,
        )
        topic_distribution = TopicDistribution(
            window         = tc.topic_distribution.window,
            stride         = tc.topic_distribution.stride,
            min_similarity = tc.topic_distribution.min_similarity,
            batch_size     = tc.topic_distribution.batch_size,
        )
        report_nr = (
            config.topic_report.sections.topics.n_repr_docs_per_topic
            if config.topic_report is not None else 5
        )
        llm_nr = config.llm.n_repr_docs_for_labeling if config.llm is not None else 3
        return cls(
            doc_dataset          = tc.doc_dataset,
            allow_abbrev         = tc.berteley.allow_abbrev,
            distance             = tc.distance,
            bertopic_model       = bertopic_model,
            topic_distribution   = topic_distribution,
            nr_repr_docs         = max(report_nr, llm_nr),
            export_dir           = tc.export.export_dir,
            n_neighbors          = tc.umap.n_neighbors,
            n_components         = tc.umap.n_components,
            min_topic_size_range = tc.hdbscan.min_topic_size_range,
            min_sample_range     = tc.hdbscan.min_sample_range,
            topic_size_step      = tc.hdbscan.topic_size_step,
            min_sample_step      = tc.hdbscan.min_sample_step,
            keep_n_results       = tc.keep_n_results,
            ranking_scorer       = tc.coherence_scorer.ranking,
            purity_scorer        = tc.coherence_scorer.purity,
            run_name             = tc.export.run_name,
            labeler_config       = config.llm,
            best_model_index     = tc.best_model_index,
        )

    # ---- runtime --------------------------------------------------------

    def _clean_dataset(self, dataset, show_progress):
        return clean_dataset(dataset, self.allow_abbrev, show_progress)

    def _make_run_dir(self) -> Path:
        """Create a unique subdirectory under ``export_dir`` for this run.
        Raises FileExistsError if the directory already exists, unless
        ``overwrite=True`` was set at construction time."""
        name = self.run_name or datetime.now().strftime('%Y-%m-%dT%H%M%S')
        run_dir = Path(self.export_dir) / name
        run_dir.mkdir(parents=True, exist_ok=self.overwrite)
        self._run_dir = str(run_dir)
        return run_dir

    def _label_selected_model(self) -> None:
        """Generate and cache LLM labels for the model at best_model_index."""
        from pysyrev.core.llm import label_topics
        from pysyrev.core.report_data import find_best_results_csv, build_file_prefix

        csv_path, distance_name = find_best_results_csv(self._run_dir)
        best_results = pd.read_csv(csv_path)
        if self.best_model_index >= len(best_results):
            print(f"[TopicModel] best_model_index={self.best_model_index} out of range — skipping label generation")
            return
        row = best_results.iloc[self.best_model_index]
        file_prefix = build_file_prefix(row["hdbscan"], row["umap"], distance_name)

        if load_cached_labels(self._run_dir, file_prefix) is not None:
            print(f"[TopicModel] Labels already cached for model {self.best_model_index} — skipping")
            return

        tinfo_path = os.path.join(self._run_dir, "topic_info", f"{file_prefix}.csv")
        topic_info = pd.read_csv(tinfo_path)
        print(f"[TopicModel] Generating topic labels for model {self.best_model_index}…")
        labels = label_topics(topic_info, self.labeler_config)
        path = save_labels(self._run_dir, file_prefix, labels)
        print(f"[TopicModel] Topic labels saved to: {path}")

    def run(self, dataset: pd.DataFrame = None, show_progress=True):
        """Run the topic modelling pipeline.

        Parameters
        ----------
        dataset : pd.DataFrame, optional
            Reviewed-included dataset. If None, loaded from ``doc_dataset``
            (set via config or auto-detected by Config.load).
        show_progress : bool
        """
        run_dir = self._make_run_dir()

        if dataset is None:
            if not self.doc_dataset:
                raise ValueError(
                    "No dataset provided: pass a DataFrame to run() or set "
                    "doc_dataset in the topic_model section of your config."
                )
            dataset = pd.read_csv(self.doc_dataset)
        cleans_docs, surviving_indices = self._clean_dataset(dataset, show_progress=show_progress)
        print(
            f"[TopicModel] {len(dataset)} documents in dataset → "
            f"{len(cleans_docs)} survived preprocessing."
        )
        embeddings = self.bertopic_model.embedding_model.encode(
            cleans_docs, show_progress_bar=show_progress,
        )
        result = topic_modeling(
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
            str(run_dir),
            self.nr_repr_docs,
            self.distance,
            self.ranking_scorer,
            self.purity_scorer,
            self.keep_n_results,
            show_progress,
            surviving_indices,
        )
        if self.labeler_config is not None:
            self._label_selected_model()
        return result
