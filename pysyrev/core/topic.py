import numpy as np

from berteley.preprocessing import preprocess as berteley_preprocess
from tqdm import tqdm


def clean_dataset(dataset, allow_abbrev, show_progress):

    abstract_corpus = np.asarray(dataset["abstract"])
    title_corpus = np.asarray(dataset["title"])

    raw_documents = [". ".join(ds) for ds in zip(*[title_corpus, abstract_corpus])]

    return berteley_preprocess(raw_documents,
                               allow_abbrev=allow_abbrev,
                               show_progress=show_progress)


def topic_modeling(bertopic_model,
                   umap_n_neighbors,
                   umap_n_components,
                   min_topic_size_range,
                   min_samples_range,
                   topic_size_step,
                   min_samples_step,
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

            for min_topic_size in range(min_topic_size_range[0],
                                             min_topic_size_range[1] + 1,
                                             topic_size_step):

                for min_samples in range(min_samples_range[0],
                                         min_samples_range[1] + 1,
                                         min_samples_step):

                    if show_progress:
                        pg.update(1)
                        
    bt_model = bertopic_model(min_topic_size,
                              min_samples,
                              n_neighbors,
                              n_components)