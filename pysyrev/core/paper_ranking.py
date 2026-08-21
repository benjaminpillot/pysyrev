"""
Three-axis paper ranking — which papers of each cluster are most worth reading.

Ported from the ``cluster-top-papers-3axis`` skill. Ranks papers *within* each
cluster (here: each BERTopic topic) by an equal-weighted composite of three
axes, every sub-metric turned into a within-cluster percentile (0-1, higher =
better) so the different scales are comparable and outlier-robust:

  1. **centrality**       = mean( pct(coupling PageRank), pct(weighted degree) )
                            — position in the bibliographic-coupling network.
  2. **relevance**        = mean( pct(citations/year), pct(raw citations) )
                            — age-normalised and absolute citation impact.
  3. **representativeness** = pct( thematic typicality )
                            — cosine of the paper's TF-IDF vector to its cluster
                              centroid (how prototypical its wording is).

Citations/year uses complete years only (the incomplete current year is dropped
from the denominator). A current-year paper has no complete citation year, so it
is scored on centrality + representativeness only (relevance renormalised out),
with ``relevance = None`` and ``relevance_dropped = True`` in the output.

Axes are combined by arithmetic ``mean`` (compensatory, default), ``gmean`` (a
weak axis drags the score down), or ``chebyshev`` (maximin — ranks on the
weakest axis, rewarding all-round papers).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np


def pctl_within(idx: Sequence[int], vals) -> dict:
    """Percentile rank (0-1) of ``vals`` restricted to indices ``idx`` → {gi: pct}."""
    from scipy.stats import rankdata
    sub = np.array([vals[i] for i in idx], dtype=float)
    if len(sub) > 1:
        r = (rankdata(sub, method="average") - 1) / (len(sub) - 1)
    else:
        r = np.array([1.0] * len(sub))
    return {gi: float(r[k]) for k, gi in enumerate(idx)}


def thematic_typicality(records, labels, text_of: Optional[Callable] = None,
                        max_features: int = 4000, ngram_range=(1, 2),
                        min_df: int = 3) -> np.ndarray:
    """Cosine similarity of each record's TF-IDF vector to its cluster centroid.

    Returns an array aligned to *records* (0 for tail records / when the corpus
    is too small to vectorise). Degrades gracefully on small corpora by falling
    back to ``min_df=1`` and, failing that, all-zeros.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if text_of is None:
        def text_of(r):
            return ((r.get("title") or "") + " " + (r.get("_abstract") or "")).strip()

    texts = [text_of(r) for r in records]
    typ = np.zeros(len(records))

    X = None
    for md in (min_df, 1):
        try:
            X = TfidfVectorizer(stop_words="english", max_features=max_features,
                                ngram_range=ngram_range, min_df=md).fit_transform(texts)
        except ValueError:
            X = None
            continue
        if X.shape[1] > 0:
            break
        X = None
    if X is None or X.shape[1] == 0:
        return typ

    labels = np.asarray(labels)
    for cl in sorted(set(int(l) for l in labels if l >= 0)):
        idx = [i for i in range(len(records)) if labels[i] == cl]
        centroid = np.asarray(X[idx].mean(axis=0))
        sims = cosine_similarity(X[idx], centroid).ravel()
        for k, i in enumerate(idx):
            typ[i] = float(sims[k])
    return typ


def aggregate_axes(vals, wts, mode: str = "mean", cheby_rho: float = 0.05) -> float:
    """Combine per-axis percentile scores (each 0-1, higher = better) into one
    composite. ``mode``: ``"mean"`` (compensatory), ``"gmean"`` (weak axis
    penalised), ``"chebyshev"`` (maximin — governed by the weakest axis, a small
    mean term breaks ties). Higher = better under every mode."""
    v = np.asarray(vals, dtype=float)
    a = np.asarray(wts, dtype=float)
    s = a.sum()
    a = a / s if s > 0 else np.ones_like(v) / len(v)
    if mode == "mean":
        return float((a * v).sum())
    if mode == "gmean":
        return float(np.exp((a * np.log(np.clip(v, 1e-9, None))).sum()))
    if mode == "chebyshev":
        return float(v.min() + cheby_rho * v.mean())
    raise ValueError(f"unknown aggregate mode: {mode!r}")


def top_papers_3axis(records, labels, W, n: int = 3, weights=None,
                     current_year: Optional[int] = None, typicality=None,
                     relevance_mode: str = "blend", drop_current_year: bool = True,
                     split_current_year: bool = False,
                     aggregate: str = "mean", text_of: Optional[Callable] = None,
                     year_key: str = "publication_year",
                     cite_key: str = "cited_by_count") -> dict:
    """Top-N papers per cluster by the three-axis composite.

    Parameters
    ----------
    records : sequence of dicts (rows), aligned to *W* and *labels*.
    labels  : cluster id per record (``>= 0``; ``-1`` = tail, skipped).
    W       : symmetric coupling matrix (``len(records)`` square) — used for both
              PageRank and weighted degree.
    n       : papers kept per cluster (best first).
    weights : ``(w_centrality, w_representativeness, w_relevance)`` axis weights,
              renormalised to sum to 1. ``None`` → equal (1/3 each). For a current-year
              paper the relevance weight is dropped and the other two renormalised.
    current_year : reference year for the citations/year denominator and for detecting
              current-year papers. ``None`` → the largest *year_key* value in *records*.
    typicality : optional precomputed representativeness array (cosine-to-centroid),
              aligned to *records*. ``None`` → computed via :func:`thematic_typicality`.
    relevance_mode : ``"blend"`` = mean(pct cit/yr, pct raw citations); ``"cpy"``
              = citations-per-year alone.
    drop_current_year : when True (default) the current (incomplete) year is removed
              from the citations/year denominator, so a paper is divided by its number
              of *complete* elapsed years; when False the running year is kept in the
              denominator and every paper is scored on all three axes.
    split_current_year : when True, current-year papers (the ones whose relevance axis
              is dropped) are returned in a SEPARATE ``"fronts"`` structure rather than
              mixed into the per-cluster ranking, and each cohort is truncated to *n*
              independently. This separates the established literature (scored on all
              three axes) from the in-progress research fronts (centrality + typicality
              only), instead of ranking a 2-axis paper against 3-axis ones.
    aggregate : ``"mean"`` | ``"gmean"`` | ``"chebyshev"`` (see :func:`aggregate_axes`).
    text_of : callable ``record -> str`` giving the text used for thematic typicality
              (``None`` → ``title`` + ``_abstract``). Ignored when *typicality* is given.
    year_key, cite_key : record keys for the publication year and citation count
              (defaults ``"publication_year"`` / ``"cited_by_count"``).

    Returns
    -------
    dict
        With *split_current_year* False (default): ``{cluster: [row, ...]}`` where
        each row is a dict with ``index, score, centrality, relevance,
        representativeness, relevance_dropped, cited_by_count, cites_per_year``,
        sorted best-first. For a current-year paper ``relevance`` and
        ``cites_per_year`` are ``None`` and ``relevance_dropped`` is ``True``.
        With *split_current_year* True: ``{"historical": {cluster: [...]},
        "fronts": {cluster: [...]}}`` — same row dicts, current-year papers moved
        to ``"fronts"``.
    """
    import igraph as ig

    W = np.asarray(W, dtype=float)
    if current_year is None:
        current_year = max((r.get(year_key) or 0) for r in records)

    # centrality sub-metrics from the coupling graph
    srcs, dsts = np.triu(W, 1).nonzero()
    gw = ig.Graph(n=len(W), edges=list(zip(srcs.tolist(), dsts.tolist())))
    if len(srcs):
        gw.es["weight"] = [float(W[i, j]) for i, j in zip(srcs, dsts)]
        pr = np.array(gw.pagerank(weights="weight"))
        wdeg = np.array(gw.strength(weights="weight"))
    else:
        pr = np.zeros(len(W))
        wdeg = np.zeros(len(W))

    # relevance sub-metrics
    raw = np.array([r.get(cite_key) or 0 for r in records], dtype=float)
    eff_year = (current_year - 1) if drop_current_year else current_year
    years_elapsed = np.array(
        [eff_year - (r.get(year_key) or current_year) + 1 for r in records], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cyr = np.where(years_elapsed >= 1, raw / years_elapsed, np.nan)
    no_relevance = years_elapsed < 1

    if typicality is None:
        typicality = thematic_typicality(records, labels, text_of=text_of)
    typicality = np.asarray(typicality, dtype=float)

    labels = np.asarray(labels)
    if weights is None:
        weights = (1.0, 1.0, 1.0)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    out, out_hist, out_front = {}, {}, {}
    for cl in sorted(set(int(l) for l in labels if l >= 0)):
        idx = [i for i in range(len(records)) if labels[i] == cl]
        p_pr, p_wd = pctl_within(idx, pr), pctl_within(idx, wdeg)
        p_ty = pctl_within(idx, typicality)
        idx_rel = [gi for gi in idx if not no_relevance[gi]]
        p_cy = pctl_within(idx_rel, cyr) if idx_rel else {}
        p_rw = pctl_within(idx_rel, raw) if idx_rel else {}
        rows = []
        for gi in idx:
            cent = (p_pr[gi] + p_wd[gi]) / 2
            rep = p_ty[gi]
            if no_relevance[gi]:
                score = aggregate_axes([cent, rep], [w[0], w[1]], mode=aggregate)
                rel = None
            else:
                rel_val = p_cy[gi] if relevance_mode == "cpy" else (p_cy[gi] + p_rw[gi]) / 2
                score = aggregate_axes([cent, rep, rel_val], [w[0], w[1], w[2]], mode=aggregate)
                rel = float(rel_val)
            rows.append(dict(index=gi, score=float(score),
                             centrality=float(cent), relevance=rel,
                             representativeness=float(rep),
                             relevance_dropped=bool(no_relevance[gi]),
                             cited_by_count=int(raw[gi]),
                             cites_per_year=(None if no_relevance[gi] else float(cyr[gi]))))
        rows.sort(key=lambda d: d["score"], reverse=True)
        if split_current_year:
            # partition BEFORE truncation so each cohort gets its own top-n
            out_hist[cl]  = [r for r in rows if not r["relevance_dropped"]][:n]
            out_front[cl] = [r for r in rows if r["relevance_dropped"]][:n]
        else:
            out[cl] = rows[:n]
    if split_current_year:
        return {"historical": out_hist, "fronts": out_front}
    return out
