import os.path

import numpy as np
import pandas as pd

from berteley.preprocessing import preprocess as berteley_preprocess
from berteley.models import fit as berteley_fit
from tqdm import tqdm


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
    purity = list()
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
                                              min_samples,
                                              n_neighbors,
                                              n_components)

                    topics, probs, topic_sizes, topic_model, topic_words, metrics = berteley_fit(documents,
                                                                                                 reduced_embeddings,
                                                                                                 **bt_model)
                    # Model
                    topic_models.append(topic_model)
                    nb_topics.append(len(topics) - 1)

                    # Compute metrics
                    try:
                        nb_outliers.append(topic_sizes[-1])
                    except KeyError:
                        nb_outliers.append(np.nan)
                    p_i_log_p_i = topics["Count"].values[1::] / nb_documents * np.log(
                        topics["Count"].values[1::] / nb_documents)
                    entropy.append(-1 * np.sum(p_i_log_p_i))
                    purity.append(metrics["Coherence"] * metrics["Diversity"])

    #---------
    # Rank models according to entropy, purity, and nb of outliers
    entropy = np.asarray(entropy)
    purity = np.asarray(purity)
    nb_outliers = np.asarray(nb_outliers)
    no_outliers = nb_documents - nb_outliers
    normalized_entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
    normalized_purity = (purity - np.min(purity)) / (np.max(purity) - np.min(purity))
    normalized_outliers = (no_outliers - np.min(no_outliers)) / (np.max(no_outliers) - np.min(no_outliers))

    # Distance to ideal
    if distance_name == "euclidean":
        distance = np.sqrt((1 - normalized_entropy)**2 + (1 - normalized_purity) ** 2 + (1 - normalized_outliers)**2)
    else:  # Chebyshev
        distance = np.maximum(1 - normalized_entropy, 1 - normalized_outliers, 1 - normalized_purity)

    results = pd.DataFrame({
        "hdbscan"     : hdbscan,
        "umap"        : umap,
        "nb_topics"   : nb_topics,
        "entropy"     : entropy,
        "purity"      : purity,
        "nb_outliers" : nb_outliers,
        "distance"    : distance,
        "model"       : topic_models
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

    # Store results
    for hdbscan_, umap_, model in zip(best_results.hdbscan,
                                      best_results.umap,
                                      best_results.model):

        # Output file names
        file_name = f"hdbscan={hdbscan_}_umap={umap_}.csv"
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
        out_docs = pd.concat([pd.DataFrame({"document": documents}),
                              dataset,
                              pd.DataFrame({"topic": model.topics_})], axis=1)
        bertopic_results = pd.concat([out_docs,
                                      topic_distribution_df], axis=1)
        bertopic_results.to_csv(out_file["bertopic_results"])

        # Topic info for N representative documents
        topic_info = model.get_topic_info()
        repr_docs, _, _, id_ = (
            model.extract_representative_docs(c_tf_idf=model.c_tf_idf,
                                              documents=out_docs,
                                              topics=model.topic_representations_,
                                              nr_repr_docs=nr_repr_docs))

        for field in dataset.columns:
            topic_info[f"repr_doc_{field}"] = [dataset[field].values[np.asarray(idx)] for idx in id_]

        topic_info.to_csv(out_file["topic_info"])

                    ###############
                    # Store metrics
                    # with open(metrics_file, "a") as file:
                    #     try:
                    #         nb_outliers = topic_sizes[-1]
                    #     except KeyError:
                    #         nb_outliers = np.nan
                    #     try:
                    #         print(f"Min topic size = {min_topic_size}, Min samples = {min_samples} :"
                    #               f"{metrics}"
                    #               f", nb of outliers = {nb_outliers}",
                    #               file=file
                    #               )
                    #     except KeyError:
                    #         pass

    if show_progress:
        pg.close()

    return 0