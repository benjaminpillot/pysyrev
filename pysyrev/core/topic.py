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


def topic_entropy(topic_sizes) -> float:
    """Shannon entropy (in nats) of the distribution of *topic_sizes*.

    The distribution is over the clustered documents: an outlier belongs to no
    topic, and outliers already have a ranking criterion of their own (the
    coverage term). Taking the whole corpus as the denominator instead folds
    them into this metric a second time — for a clustered fraction ``f`` it
    yields ``f * (H + ln(1/f))`` rather than ``H`` — which leaves two of the
    four criteria measuring the same thing, and makes a model look unbalanced
    for having outliers rather than for having lopsided topics.
    """
    sizes = np.asarray(list(topic_sizes), dtype=float)
    total = sizes.sum()
    if sizes.size == 0 or total == 0:
        return 0.0
    p = sizes / total
    return float(-np.sum(p * np.log(p)))


def rank_by_distance(results, distance_name, nb_documents, coherence_col):
    """Add the ``distance`` column to *results* and sort it best-first.

    Ranking has to come after every filter that removes candidates, because
    :func:`_compute_distance` min-max normalizes each metric: the models present
    here set the scale the others are scored against. Ranking the whole grid and
    filtering afterwards leaves the survivors scored against models that are no
    longer in contention — on a real run, out-of-band models (a degenerate
    2-topic fit, say) stretched the entropy scale enough to hand every one of the
    25 best slots to the smallest topic count in the band.
    """
    distance = _compute_distance(distance_name,
                                 results["entropy"].to_numpy(),
                                 results[coherence_col].to_numpy(),
                                 nb_documents - results["nb_outliers"].to_numpy(),
                                 results["diversity"].to_numpy())
    results = results.copy()
    position = (results.columns.get_loc("model") if "model" in results.columns
                else len(results.columns))
    results.insert(position, "distance", distance)
    return results.sort_values(by="distance", ascending=True)


_META_COLS = ["id", "year", "cited_by", "document_type", "title", "doi"]

# How many times a run may re-aim its min_topic_size grid at the desired band.
# Each attempt costs another HDBSCAN sweep, and the estimate it is aimed with
# converges fast — a second attempt only matters when the first overshot.
_MAX_REAIM_ATTEMPTS = 2


def derive_min_topic_size_range(n_docs, topic_range, n_steps=8):
    """Derive an HDBSCAN ``min_topic_size`` grid aimed at a desired topic band.

    HDBSCAN's ``min_cluster_size`` (our ``min_topic_size``) trades off inversely
    against the number of topics: bigger clusters -> fewer topics. A first-order
    estimate is ``min_cluster_size ~= n_docs / nb_topics``, so a desired band
    ``[t_min, t_max]`` maps to a ``min_topic_size`` range ``[n_docs / t_max,
    n_docs / t_min]``.

    That estimate is biased: it reads ``min_cluster_size`` as the average topic
    size, where HDBSCAN treats it as a floor, and it counts outlier documents as
    if they joined a topic. Both push the real topic count below the target, so
    the grid can land entirely above the band. It is an opening guess;
    :func:`reaim_min_topic_size_range` corrects it from measurements.

    Returns ``([lo, hi], step)`` with ``lo >= 2`` and ``step >= 1``, the grid
    spanning roughly *n_steps* values.
    """
    t_min, t_max = topic_range
    lo = max(2, round(n_docs / t_max))
    hi = max(lo, round(n_docs / t_min))
    step = max(1, round((hi - lo) / max(1, n_steps - 1)))
    return [lo, hi], step


def min_topic_sizes_of(results):
    """The ``min_topic_size`` of each fitted model, read back from ``hdbscan``."""
    return results["hdbscan"].str.split("_").str[0].astype(int)


def _topic_counts_seen(results):
    """The distinct topic counts a sweep produced, as plain ints for printing."""
    return sorted(int(n) for n in results["nb_topics"].unique())


def reaim_min_topic_size_range(results, topic_range, n_steps=8):
    """A ``min_topic_size`` grid calibrated on the models already fitted.

    Returns the list of ``min_topic_size`` values to sweep next, or None when the
    results carry too little signal to re-aim from.

    The topic count falls roughly as the inverse of ``min_cluster_size``, so
    ``nb_topics * min_topic_size`` is approximately constant for a given corpus
    and every fitted model measures it. Inverting that constant aims the next
    grid at the band.

    Each size contributes its *best* count, not its mean: ``min_samples`` and the
    UMAP parameters vary alongside it and can only take the count down from what
    a given size allows.
    """
    observed = pd.DataFrame({"size": min_topic_sizes_of(results),
                             "nb_topics": results["nb_topics"]})
    best = observed[observed["nb_topics"] > 0].groupby("size")["nb_topics"].max()
    if best.empty:
        return None
    k = float(np.median(best.index.to_numpy() * best.to_numpy()))
    if k <= 0:
        return None

    t_min, t_max = topic_range
    lo = max(2, int(round(k / t_max)))
    hi = max(lo, int(round(k / t_min)))
    step = max(1, int(round((hi - lo) / max(1, n_steps - 1))))
    return list(range(lo, hi + 1, step))


def min_samples_of(results):
    """The ``min_samples`` of each fitted model, read back from ``hdbscan``."""
    return results["hdbscan"].str.split("_").str[1].astype(int)


def report_unused_min_samples(best_results, fitted_results):
    """Say when the selected models never reached into the min_samples grid.

    ``min_samples`` cannot be derived the way ``min_topic_size`` can: its
    first-order effect is on the outlier fraction, and what it does to the topic
    count is indirect and corpus-dependent. So the grid is swept and the ranking
    decides — which is exactly why a range set too wide can only be recognised
    afterwards, once a third of the fits have gone to a corner of the search that
    won nothing.

    This reports rather than presumes. It measures against what was actually
    fitted, not the configured range, since the degenerate pairs are never fitted
    in the first place; and it stays quiet unless a clear part of the range went
    unused, leaving the next run's range to the caller.
    """
    if best_results.empty:
        return
    fitted = sorted(set(min_samples_of(fitted_results)))
    if len(fitted) < 3:
        return
    lo, hi = fitted[0], fitted[-1]
    used = int(min_samples_of(best_results).max())
    # The top third of what was swept: a winner sitting one step short of the
    # ceiling is not a finding, a whole unused corner is.
    if used >= hi - (hi - lo) / 3:
        return
    print(
        f"[topic_modeling] None of the {len(best_results)} selected models used a "
        f"min_samples above {used}, though the sweep went up to {hi}. Those fits "
        f"won nothing here — consider min_sample_range [{lo}, {used}] next run."
    )


def has_in_band(results, topic_range):
    """Whether any fitted model's topic count falls inside *topic_range*."""
    if topic_range is None:
        return True
    lo, hi = topic_range
    return bool(((results["nb_topics"] >= lo) & (results["nb_topics"] <= hi)).any())


def filter_to_topic_band(results, topic_range):
    """Keep only rows of *results* whose ``nb_topics`` is within *topic_range*.

    This is the guarantee behind ``desired_topics``: the grids merely aim at the
    band, this filter enforces it. Reaching it here empty-handed means the band
    survived both the derived grid and the re-aimed ones, so warn and return
    *results* unchanged — the run still produces output.
    """
    if topic_range is None:
        return results
    lo, hi = topic_range
    in_band = results[(results["nb_topics"] >= lo) & (results["nb_topics"] <= hi)]
    if in_band.empty:
        print(
            f"[topic_modeling] Warning: no model produced a topic count in "
            f"[{lo}, {hi}] (got {_topic_counts_seen(results)}), even "
            f"after re-aiming the min_topic_size grid. Keeping the best-ranked "
            f"models regardless — widen desired_topics, or accept that a corpus "
            f"this size does not split into that many topics."
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

    min_topic_size_values = list(range(min_topic_size_range[0],
                                       min_topic_size_range[1] + 1,
                                       topic_size_step))

    min_samples_values = list(range(min_samples_range[0],
                                    min_samples_range[1] + 1,
                                    min_samples_step))

    # UMAP is the expensive step and depends on none of the HDBSCAN parameters,
    # so each reduction is computed once and reused by every sweep — including
    # the re-aimed ones, which is what makes re-aiming affordable.
    reductions = {}

    def reduce_embeddings(n_neighbors, n_components):
        key = (n_neighbors, n_components)
        if key not in reductions:
            reductions[key] = bertopic_model.umap_model(n_neighbors, n_components,
                                                        embeddings)
        return reductions[key]

    def grid_pairs(size_values):
        """The ``(min_topic_size, min_samples)`` pairs worth fitting.

        A ``min_samples`` above ``min_cluster_size`` is a degenerate corner of
        HDBSCAN: the density threshold is stricter than the cluster it is meant
        to keep, so most documents fall out as noise and the fit can only return
        fewer topics than the same size with a smaller ``min_samples``. On a wide
        min_sample_range those pairs are most of the grid and none of them can
        move a run toward its desired band.

        Dropping them must not drop the size itself. A size below the whole
        min_samples grid would otherwise vanish from the sweep, and those are the
        small sizes — exactly the ones a re-aim reaches for when it needs more
        topics. Such a size is fitted once, at the least conservative
        ``min_samples`` on offer.
        """
        smallest = min(min_samples_values)
        pairs = []
        for size in size_values:
            usable = [ms for ms in min_samples_values if ms <= size]
            pairs.extend((size, ms) for ms in (usable or [smallest]))
        return pairs

    def sweep(size_values, desc):
        """Fit every (umap, min_topic_size, min_samples) combination of a grid."""
        pairs = grid_pairs(size_values)
        rows = []
        pg = (tqdm(total=len(umap_n_neighbors) * len(umap_n_components) * len(pairs),
                   desc=desc) if show_progress else None)

        for n_neighbors in umap_n_neighbors:
            for n_components in umap_n_components:
                reduced_embeddings = reduce_embeddings(n_neighbors, n_components)

                for min_topic_size, min_samples in pairs:
                    if pg is not None:
                        pg.update(1)

                    bt_model = bertopic_model(min_topic_size, min_samples)
                    topics, probs, topic_sizes, topic_model, topic_words, metrics = berteley_fit(
                        documents,
                        reduced_embeddings,
                        coherence_scorer=ranking_scorer,
                        **bt_model)

                    # Outliers are keyed on -1 and are not a topic. Taking them
                    # out first leaves the real topics to be counted, and a fit
                    # that produced none is a plain zero rather than the NaN that
                    # would poison every model's normalised distance.
                    sizes = dict(topic_sizes)
                    outliers = sizes.pop(-1, 0)

                    rows.append({
                        "hdbscan":      f"{min_topic_size}_{min_samples}",
                        "umap":         f"{n_neighbors}_{n_components}",
                        "nb_topics":    len(sizes),
                        "entropy":      topic_entropy(sizes.values()),
                        ranking_scorer: metrics["Coherence"],
                        "diversity":    metrics["Diversity"],
                        "nb_outliers":  outliers,
                        "model":        topic_model,
                    })

        if pg is not None:
            pg.close()
        return pd.DataFrame(rows)

    results = sweep(min_topic_size_values, "Topic modeling in process...")

    # `desired_topics` promises a topic count, and the opening grid is derived
    # from a rule of thumb that can place the whole grid on the wrong side of the
    # band — every model out of band, and no step through that grid able to reach
    # it. The models just fitted measure what the rule of thumb guessed, so
    # re-aim from them and sweep again rather than reporting a miss.
    swept = {tuple(min_topic_size_values)}
    for attempt in range(1, _MAX_REAIM_ATTEMPTS + 1):
        if has_in_band(results, topic_range):
            break
        size_values = reaim_min_topic_size_range(
            results, topic_range, n_steps=len(min_topic_size_values))
        if size_values is None or tuple(size_values) in swept:
            break
        swept.add(tuple(size_values))
        print(
            f"[topic_modeling] No model in the desired band {list(topic_range)} "
            f"(got {_topic_counts_seen(results)}). Re-aiming min_topic_size at "
            f"{size_values[0]}–{size_values[-1]} and sweeping again "
            f"({attempt}/{_MAX_REAIM_ATTEMPTS})…"
        )
        results = pd.concat(
            [results, sweep(size_values, f"Topic modeling (re-aimed grid {attempt})...")],
            ignore_index=True)

    #---------
    # Rank models according to entropy, coherence, and nb of outliers
    # Keep only models whose topic count falls in the desired band (if set),
    # *then* rank: the ranking is relative to the models still in contention.
    fitted_results = results
    results = filter_to_topic_band(results, topic_range)
    results = rank_by_distance(results, distance_name, nb_documents, ranking_scorer)

    best_results = results.iloc[:keep_n_results]
    report_unused_min_samples(best_results, fitted_results)

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

    # The progress bar belongs to `sweep`, which opens and closes one per grid.

    return 0