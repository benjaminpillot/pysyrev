import os.path

import numpy as np
import pandas as pd

from pysyrev.core.berteley.preprocessing import preprocess as berteley_preprocess
from pysyrev.core.berteley.models import fit as berteley_fit, _calculate_metrics
from tqdm import tqdm


def _compute_distance(distance, *values):
    norm = []
    for n in values:
        r = np.max(n) - np.min(n)
        # When all values are identical the metric doesn't discriminate: treat as ideal (norm=1).
        norm.append((n - np.min(n)) / r if r > 0 else np.ones_like(n, dtype=float))

    if distance == "euclidean":
        # Use Python sum (not np.sum) to accumulate arrays element-wise across metrics.
        # np.sum on a list of arrays collapses to a scalar, giving the same distance for every model.
        return np.sqrt(sum((1 - norm_n) ** 2 for norm_n in norm))
    else:  # Chebyshev
        return np.maximum.reduce([1 - norm_n for norm_n in norm])


_META_COLS = ["id", "year", "cited_by", "document_type", "title", "doi"]


def derive_min_topic_size_range(n_docs, topic_range, n_steps=8):
    """Derive an HDBSCAN ``min_topic_size`` grid aimed at a desired topic band.

    HDBSCAN's ``min_cluster_size`` (our ``min_topic_size``) trades off inversely
    against the number of topics: bigger clusters -> fewer topics. A first-order
    estimate is ``min_cluster_size ~= n_docs / nb_topics``, so a desired band
    ``[t_min, t_max]`` maps to a ``min_topic_size`` range ``[n_docs / t_max,
    n_docs / t_min]``. This only *aims* the grid; the exact band is enforced
    afterwards by filtering results (see ``topic_modeling``'s ``topic_range``),
    because outliers and UMAP structure make the mapping approximate.

    Returns ``([lo, hi], step)`` with ``lo >= 2`` and ``step >= 1``, the grid
    spanning roughly *n_steps* values.
    """
    t_min, t_max = topic_range
    lo = max(2, round(n_docs / t_max))
    hi = max(lo, round(n_docs / t_min))
    step = max(1, round((hi - lo) / max(1, n_steps - 1)))
    return [lo, hi], step


def filter_to_topic_band(results, topic_range):
    """Keep only rows of *results* whose ``nb_topics`` is within *topic_range*.

    This is the guarantee behind ``desired_topics``: the derived grid merely aims
    at the band, this filter enforces it. If nothing lands in-band (band
    unreachable with the explored grid), warn and return *results* unchanged so
    the run still produces output.
    """
    if topic_range is None:
        return results
    lo, hi = topic_range
    in_band = results[(results["nb_topics"] >= lo) & (results["nb_topics"] <= hi)]
    if in_band.empty:
        print(
            f"[topic_modeling] Warning: no model produced a topic count in "
            f"[{lo}, {hi}] (got {sorted(results['nb_topics'].unique())}). "
            f"Keeping the best-ranked models regardless — widen desired_topics "
            f"or adjust the corpus/params."
        )
        return results
    return in_band


def clean_dataset(dataset, allow_abbrev, show_progress):
    abstract_corpus = np.asarray(dataset["abstract"])
    title_corpus = np.asarray(dataset["title"])
    raw_documents = [". ".join(ds) for ds in zip(*[title_corpus, abstract_corpus])]
    return berteley_preprocess(raw_documents,
                               allow_abbrev=allow_abbrev,
                               show_progress=show_progress,
                               return_indices=True)


def topic_modeling(dataset,
                   documents,
                   bertopic_model,
                   topic_distribution,
                   embeddings,
                   umap_n_neighbors,
                   umap_n_components,
                   min_topic_size_range,
                   min_samples_range,
                   topic_size_step,
                   min_samples_step,
                   export_to,
                   nr_repr_docs,
                   distance_name,
                   ranking_scorer,
                   purity_scorer,
                   keep_n_results,
                   show_progress,
                   surviving_indices=None,
                   topic_range=None):

    # cluster_sel_method = bertopic_model.hdbscan_model.clusterselection_method

    nb_documents = len(documents)

    # UMAP needs strictly more samples than n_components; filter invalid values early.
    valid_n_components = [nc for nc in umap_n_components if nc < nb_documents]
    if not valid_n_components:
        raise ValueError(
            f"topic_modeling requires more preprocessed documents than the largest "
            f"n_components value. Got {nb_documents} document(s) after preprocessing "
            f"but n_components = {sorted(umap_n_components)}. "
            "Increase your corpus size (e.g. raise review sample_size or use the full "
            "dataset), or lower n_components."
        )
    if len(valid_n_components) < len(umap_n_components):
        skipped = sorted(set(umap_n_components) - set(valid_n_components))
        print(
            f"[topic_modeling] Warning: skipping n_components {skipped} — "
            f"not enough documents ({nb_documents})."
        )
    umap_n_components = valid_n_components

    min_topic_size_values = range(min_topic_size_range[0],
                                  min_topic_size_range[1] + 1,
                                  topic_size_step)

    min_samples_values = range(min_samples_range[0],
                               min_samples_range[1] + 1,
                               min_samples_step)

    # UMAP & HDBSCAN parameters
    umap = []
    hdbscan = []

    # Metrics
    nb_topics = []
    entropy = list()
    coherence = list()
    diversity = list()
    nb_outliers = list()
    topic_models = list()

    if show_progress:
        total_len = (len(umap_n_neighbors) * len(umap_n_components) *
                     len(list(min_topic_size_values)) * len(list(min_samples_values)))
        pg = tqdm(total=total_len, desc="Topic modeling in process...")

    for n_neighbors in umap_n_neighbors:
        for n_components in umap_n_components:

            # UMAP
            reduced_embeddings = bertopic_model.umap_model(n_neighbors, n_components, embeddings)

            for min_topic_size in min_topic_size_values:

                for min_samples in min_samples_values:

                    # Store hyperparameters
                    umap.append(f"{n_neighbors}_{n_components}")
                    hdbscan.append(f"{min_topic_size}_{min_samples}")


                    if show_progress:
                        pg.update(1)
                        
                    bt_model = bertopic_model(min_topic_size,
                                              min_samples)

                    topics, probs, topic_sizes, topic_model, topic_words, metrics = berteley_fit(documents,
                                                                                                 reduced_embeddings,
                                                                                                 coherence_scorer=ranking_scorer,
                                                                                                 **bt_model)
                    # Model
                    topic_models.append(topic_model)
                    nb_topics.append(len(topic_sizes) - 1)

                    # Compute metrics
                    try:
                        nb_outliers.append(topic_sizes.pop(-1))
                    except KeyError:
                        nb_outliers.append(np.nan)
                    values = np.asarray(list(topic_sizes.values()))
                    p_i_log_p_i = values / nb_documents * np.log(values / nb_documents)
                    entropy.append(-1 * np.sum(p_i_log_p_i))
                    coherence.append(metrics["Coherence"])
                    diversity.append(metrics["Diversity"])

    #---------
    # Rank models according to entropy, coherence, and nb of outliers
    entropy = np.asarray(entropy)
    coherence = np.asarray(coherence)
    nb_outliers = np.asarray(nb_outliers)
    distance = _compute_distance(distance_name,
                                 entropy,
                                 coherence,
                                 nb_documents - nb_outliers,
                                 diversity)

    results = pd.DataFrame({
        "hdbscan"                       : hdbscan,
        "umap"                          : umap,
        "nb_topics"                     : nb_topics,
        "entropy"                       : entropy,
        ranking_scorer                  : coherence,
        "diversity"                     : diversity,
        "nb_outliers"                   : nb_outliers,
        "distance"                      : distance,
        "model"                         : topic_models
    }).sort_values(by="distance", ascending=True)

    # Keep only models whose topic count falls in the desired band (if set).
    results = filter_to_topic_band(results, topic_range)

    best_results = results.iloc[:keep_n_results]

    # Build directory
    tinfo_dir = os.path.join(export_to, "topic_info")
    bertopic_dir = os.path.join(export_to, "bertopic_results")
    metrics_dir = os.path.join(export_to, "metrics")

    for dir_ in [export_to, tinfo_dir, bertopic_dir, metrics_dir]:
        try:
            os.makedirs(dir_)
        except FileExistsError:
            pass

    coherence_downstream = list()
    purity = list()

    if purity_scorer != ranking_scorer:
        compute_purity = True
    else:
        compute_purity = False

    # Store results
    for hdbscan_, umap_, model in zip(best_results.hdbscan,
                                      best_results.umap,
                                      best_results.model):

        # Output file names
        file_name = f"hdbscan={hdbscan_}_umap={umap_}_distance={distance_name}.csv"
        out_file = {"topic_info": os.path.join(tinfo_dir,
                                               file_name),
                    "bertopic_results": os.path.join(bertopic_dir,
                                                     file_name)}

        # Bertopic results
        topic_dist, _ = (
            model.approximate_distribution(documents,
                                           window=topic_distribution.window,
                                           stride=topic_distribution.stride,
                                           min_similarity=topic_distribution.min_similarity,
                                           batch_size=topic_distribution.batch_size))
        topic_distribution_df = pd.DataFrame(topic_dist,
                                             columns=[f"topic#{topic_n}" for
                                                      topic_n in range(topic_dist.shape[1])])

        # Prepare your documents to be used in a dataframe
        out_docs = pd.DataFrame({"Document": documents,
                                 "ID": range(len(documents)),
                                 "Topic": model.topics_})

        meta_parts = []
        if surviving_indices is not None:
            meta_df = dataset.iloc[surviving_indices].reset_index(drop=True)
            available = [c for c in _META_COLS if c in meta_df.columns]
            if available:
                meta_parts.append(meta_df[available].reset_index(drop=True))

        bertopic_results = pd.concat(
            [out_docs, topic_distribution_df] + meta_parts, axis=1
        )
        bertopic_results.to_csv(out_file["bertopic_results"],
                                index=False)

        # Topic info for N representative documents
        topic_info = model.get_topic_info()
        repr_docs, _, _, id_ = (
            model._extract_representative_docs(c_tf_idf=model.c_tf_idf_,
                                               documents=out_docs,
                                               topics=model.topic_representations_,
                                               nr_repr_docs=nr_repr_docs))
        for field in dataset.columns:
            topic_info[f"repr_doc_{field}"] = [dataset[field].values[np.asarray(idx)].tolist() for idx in id_]
        topic_info.to_csv(out_file["topic_info"],
                          index=False)

        # Re-compute coherence if necessary
        if compute_purity:
            final_metrics = _calculate_metrics(documents, model, model.topics_, purity_scorer)
            coherence_downstream.append(final_metrics["Coherence"])
            purity.append(final_metrics["Coherence"] * final_metrics["Diversity"])

    # Compute and store metrics
    if compute_purity:
        best_results.loc[:, purity_scorer] = coherence_downstream
        best_results.loc[:, "purity"] = purity
    else:
        best_results.loc[:, "purity"] = best_results.loc[:, ranking_scorer] * best_results.loc[:, "diversity"]

    best_results.loc[:, "distance_with_purity"] = _compute_distance(distance_name,
                                                                    best_results["entropy"],
                                                                    nb_documents - best_results["nb_outliers"],
                                                                    best_results["purity"])
    best_results.to_csv(os.path.join(metrics_dir, f"{distance_name}_best_results.csv"),
                        index=False)
    results.to_csv(os.path.join(metrics_dir, f"{distance_name}_results.csv"),
                   index=False)

    if show_progress:
        pg.close()

    return 0