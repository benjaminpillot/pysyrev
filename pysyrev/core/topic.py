import os.path

import numpy as np
import pandas as pd

from berteley.preprocessing import preprocess as berteley_preprocess
from berteley.models import fit as berteley_fit, _calculate_metrics
from tqdm import tqdm


def _compute_distance(distance, *values):
    norm = [(n - np.min(n)) / (np.max(n) - np.min(n)) for n in values]

    # Distance to ideal
    if distance == "euclidean":
        return np.sqrt(np.sum([(1 - norm_n) ** 2 for norm_n in norm]))
        # return np.sqrt((1 - norm[0]) ** 2 + (1 - norm[1]) ** 2 + (1 - norm[2]) ** 2)
    else:  # Chebyshev
        return np.maximum.reduce([1 - norm_n for norm_n in norm])


def clean_dataset(dataset, allow_abbrev, show_progress):

    abstract_corpus = np.asarray(dataset["abstract"])
    title_corpus = np.asarray(dataset["title"])

    raw_documents = [". ".join(ds) for ds in zip(*[title_corpus, abstract_corpus])]

    return berteley_preprocess(raw_documents,
                               allow_abbrev=allow_abbrev,
                               show_progress=show_progress)


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
                   show_progress):

    # cluster_sel_method = bertopic_model.hdbscan_model.clusterselection_method

    min_topic_size_values = range(min_topic_size_range[0],
                                  min_topic_size_range[1] + 1,
                                  topic_size_step)

    min_samples_values = range(min_samples_range[0],
                               min_samples_range[1] + 1,
                               min_samples_step)

    # Nb of documents
    nb_documents = len(documents)

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

        bertopic_results = pd.concat([out_docs,
                                      topic_distribution_df], axis=1)
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