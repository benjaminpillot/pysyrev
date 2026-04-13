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
                   embeddings,
                   umap_n_neighbors,
                   umap_n_components,
                   min_topic_size_range,
                   min_samples_range,
                   topic_size_step,
                   min_samples_step,
                   export_to,
                   cluster_sel_method,
                   window,
                   stride,
                   min_similarity,
                   batch_size,
                   nr_repr_docs,
                   show_progress):

    min_topic_size_values = range(min_topic_size_range[0],
                                  min_topic_size_range[1] + 1,
                                  topic_size_step)

    min_samples_values = range(min_samples_range[0],
                               min_samples_range[1] + 1,
                               min_samples_step)

    if show_progress:
        total_len = (len(umap_n_neighbors) * len(umap_n_components) *
                     len(list(min_topic_size_values)) * len(list(min_samples_values)))
        pg = tqdm(total=total_len, desc="Topic modeling in process...")

    for n_neighbors in umap_n_neighbors:
        for n_components in umap_n_components:

            out_dir = os.path.join(export_to,
                                   f"neighbors={n_neighbors}_components={n_components}",
                                   f"{cluster_sel_method.upper()}")
            tinfo_dir = os.path.join(out_dir, "topic_info")
            bertopic_dir = os.path.join(out_dir, "bertopic_results")
            metrics_dir = os.path.join(out_dir, "metrics")

            for dir_ in [out_dir, tinfo_dir, bertopic_dir, metrics_dir]:
                try:
                    os.makedirs(dir_)
                except FileExistsError:
                    pass

            metrics_file = os.path.join(metrics_dir,
                                        f"metrics_{cluster_sel_method}.txt")

            for min_topic_size in min_topic_size_values:

                for min_samples in min_samples_values:

                    # Output file names
                    file_name = f"topic_size={min_topic_size}_min_samples={min_samples}.csv"
                    out_file = {"topic_info": os.path.join(tinfo_dir,
                                                           file_name),
                                "bertopic_results": os.path.join(bertopic_dir,
                                                                 file_name)}

                    if show_progress:
                        pg.update(1)
                        
                    bt_model = bertopic_model(min_topic_size,
                                              min_samples,
                                              n_neighbors,
                                              n_components)

                    topics, probs, topic_sizes, topic_model, topic_words, metrics = berteley_fit(documents,
                                                                                                 embeddings,
                                                                                                 **bt_model)

                    # Bertopic results
                    topic_distr, _ = topic_model.approximate_distribution(documents,
                                                                          window=window,
                                                                          stride=stride,
                                                                          min_similarity=min_similarity,
                                                                          batch_size=batch_size)
                    topic_distribution = pd.DataFrame(topic_distr,
                                                      columns=[f"topic#{topic_n}" for
                                                               topic_n in range(topic_distr.shape[1])])
                    out_docs = pd.concat([pd.DataFrame({"document": documents}),
                                          dataset,
                                          pd.DataFrame({"topic": topic_model.topics_})], axis=1)
                    bertopic_results = pd.concat([out_docs,
                                                  topic_distribution], axis=1)
                    bertopic_results.to_csv(out_file["bertopic_results"])

                    # Topic info for N representative documents
                    topic_info = topic_model.get_topic_info()
                    repr_docs, _, _, id_ = (
                        topic_model.extract_representative_docs(c_tf_idf=topic_model.c_tf_idf,
                                                                documents=out_docs,
                                                                topics=topic_model.topic_representations_,
                                                                nr_repr_docs=nr_repr_docs))

                    for field in dataset.columns:
                        topic_info[f"repr_doc_{field}"] = [dataset[field].values[np.asarray(idx)] for idx in id_]

                    topic_info.to_csv(out_file["topic_info"])

                    ###############
                    # Store metrics
                    with open(metrics_file, "a") as file:
                        try:
                            nb_outliers = topic_sizes[-1]
                        except KeyError:
                            nb_outliers = np.nan
                        try:
                            print(f"Min topic size = {min_topic_size}, Min samples = {min_samples} :"
                                  f"{metrics}"
                                  f", nb of outliers = {nb_outliers}",
                                  file=file
                                  )
                        except KeyError:
                            pass

    if show_progress:
        pg.close()
