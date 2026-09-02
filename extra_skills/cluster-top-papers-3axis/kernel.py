"""Rank the most relevant papers per cluster using a THREE-AXIS composite indicator.

Companion to the corpus-map skill; an alternative to cluster-top-papers that
redefines the three axes and adds thematic typicality as representativeness.

Composite score = equal-weighted (1/3 each) mean of three axes, every sub-metric
converted to a WITHIN-CLUSTER percentile rank (0-1, higher = better) so the very
different scales are comparable and outlier-robust:

  1. centrality       = mean( pct(coupling PageRank), pct(weighted degree) )
                        -> both measure position in the bibliographic-coupling network:
                           PageRank = global prestige of couplings; weighted degree =
                           raw strength of shared-reference overlap.
  2. relevance        = mean( pct(citations per year), pct(raw citations) )
                        -> combines age-normalised and absolute citation impact.
  3. representativeness = pct( thematic typicality )
                        -> cosine similarity of the paper's TF-IDF vector to its
                           cluster centroid: how prototypical its wording is.

No recency guardrail by default (the relevance axis already blends raw + per-year).
Pass relevance_mode="cpy" to use citations-per-year ALONE on the relevance axis.

CURRENT-YEAR PAPERS. A paper in the current (incomplete) year has no complete year of
citation accumulation, so its relevance axis is not computable. Two ways to handle them:

  * default (split_current_year=False): current-year papers are scored on the two
    measurable axes (centrality + representativeness, weights renormalised) and ranked
    IN THE SAME per-cluster list as the historical papers, with relevance = None.

  * split_current_year=True: current-year papers are REMOVED from the main ranking and
    returned in a separate structure. The main ("historical") table then contains only
    papers with at least one complete citation year, ranked on all three axes; the
    "fronts" table contains the current-year papers ranked on centrality + typicality.
    This separates the established literature from the in-progress research fronts, which
    are not comparable on citation impact. Return shape changes (see below).
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def pctl_within(idx, vals):
    """Percentile rank (0-1) of vals restricted to indices idx -> {gi: pct}."""
    sub = np.array([vals[i] for i in idx], dtype=float)
    if len(sub) > 1:
        r = (rankdata(sub, method="average") - 1) / (len(sub) - 1)
    else:
        r = np.array([1.0] * len(sub))
    return {gi: float(r[k]) for k, gi in enumerate(idx)}


def thematic_typicality(records, labels, text_of=None,
                        max_features=4000, ngram_range=(1, 2), min_df=3):
    """Cosine similarity of each record's TF-IDF vector to its cluster centroid.

    Returns a numpy array aligned to records (0 for records with no cluster).
    labels: sequence aligned to records; cluster id >= 0, or -1 for the tail.
    """
    if text_of is None:
        def text_of(r):
            return ((r.get("title") or "") + " " + (r.get("_abstract") or "")).strip()
    texts = [text_of(r) for r in records]
    X = TfidfVectorizer(stop_words="english", max_features=max_features,
                        ngram_range=ngram_range, min_df=min_df).fit_transform(texts)
    labels = np.asarray(labels)
    typ = np.zeros(len(records))
    for cl in sorted(set(int(l) for l in labels if l >= 0)):
        idx = [i for i in range(len(records)) if labels[i] == cl]
        centroid = np.asarray(X[idx].mean(axis=0))
        sims = cosine_similarity(X[idx], centroid).ravel()
        for k, i in enumerate(idx):
            typ[i] = float(sims[k])
    return typ


def aggregate_axes(vals, wts, mode="mean", cheby_rho=0.05):
    """Combine per-axis percentile scores into one composite.

    vals : list of axis values (each 0-1, higher = better), e.g. [centrality,
           representativeness, relevance]. Any number of axes (2 for current-year).
    wts  : matching axis weights; renormalised here.
    mode : "mean"      -> weighted arithmetic mean (compensatory; default).
           "gmean"     -> weighted geometric mean (a weak axis drags the score down,
                          but cannot zero it unless it is itself ~0).
           "chebyshev" -> augmented Tchebycheff / maximin: the score is governed by the
                          WEAKEST axis (min), with a small rho*mean term to break ties
                          among papers that share the same worst axis. A paper ranks high
                          only if it is strong on EVERY axis. Non-compensatory. Axis
                          weights are ignored for the min term (equal emphasis); the
                          augmentation uses the plain mean.
    Returns a scalar; higher = better under every mode (so callers always sort descending).
    """
    import numpy as np
    v = np.asarray(vals, dtype=float)
    a = np.asarray(wts, dtype=float)
    s = a.sum()
    a = a / s if s > 0 else np.ones_like(v) / len(v)
    if mode == "mean":
        return float((a * v).sum())
    if mode == "gmean":
        return float(np.exp((a * np.log(np.clip(v, 1e-9, None))).sum()))
    if mode == "chebyshev":
        # maximin with augmentation: min axis dominates, mean breaks ties.
        return float(v.min() + cheby_rho * v.mean())
    raise ValueError(f"unknown aggregate mode: {mode!r}")


def top_papers_3axis(records, labels, W, n=3, weights=None,
                     current_year=None, typicality=None, relevance_mode="blend",
                     drop_current_year=True, split_current_year=False, aggregate="mean",
                     text_of=None, year_key="publication_year", cite_key="cited_by_count"):
    """Top-N papers per cluster by the three-axis composite.

    records   : corpus rows (dicts).
    labels    : cluster id per record (>=0, -1 = tail), aligned to records.
    W         : symmetric bibliographic-coupling matrix (n_records x n_records),
                e.g. Salton weights; used for both PageRank and weighted degree.
    typicality: optional precomputed array (else computed via thematic_typicality).
    relevance_mode : "blend" = mean(pct cit/yr, pct raw citations) [default];
                     "cpy"   = citations-per-year ALONE.
    split_current_year : if False (default) current-year papers share the per-cluster
                     ranking (scored on 2 axes, relevance=None). If True, they are
                     split out.
    aggregate : how the axes are combined into the composite -- "mean" (default,
                compensatory), "gmean" (weak axis penalised), or "chebyshev"
                (maximin: ranks on the WEAKEST axis; rewards all-round papers). See
                aggregate_axes for the exact definitions.
                Note: under "chebyshev" a current-year paper is scored on 2 axes and so
                has one fewer chance of a weak axis than a 3-axis paper; use
                split_current_year=True to rank the two cohorts separately and avoid
                comparing a 2-axis min against a 3-axis min.

    Returns:
      * split_current_year=False -> dict  cluster -> list of dicts
      * split_current_year=True  -> dict  {"historical": {cluster: [...]},
                                           "fronts":     {cluster: [...]}}
    Each row dict: {index, score, centrality, relevance, representativeness,
                    relevance_dropped, cited_by_count, cites_per_year}, sorted best-first.
    In "fronts", relevance is None and the score uses centrality + representativeness only.
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
        pr = np.zeros(len(W)); wdeg = np.zeros(len(W))

    # relevance sub-metrics
    raw = np.array([r.get(cite_key) or 0 for r in records], dtype=float)
    # citations/year over COMPLETE years only: drop the current (incomplete) year
    # from the denominator when drop_current_year=True.
    eff_year = (current_year - 1) if drop_current_year else current_year
    years_elapsed = np.array([eff_year - (r.get(year_key) or current_year) + 1 for r in records],
                             dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cyr = np.where(years_elapsed >= 1, raw / years_elapsed, np.nan)
    # papers with no complete year of accumulation (the current incomplete year, and any
    # future-dated rows): relevance is not computable -> scored on the two other axes only.
    no_relevance = years_elapsed < 1

    if typicality is None:
        typicality = thematic_typicality(records, labels, text_of=text_of)
    typicality = np.asarray(typicality, dtype=float)

    labels = np.asarray(labels)
    if weights is None:
        weights = (1.0, 1.0, 1.0)
    w = np.array(weights, dtype=float); w = w / w.sum()
    out = {}
    out_hist, out_front = {}, {}
    for cl in sorted(set(int(l) for l in labels if l >= 0)):
        idx = [i for i in range(len(records)) if labels[i] == cl]
        p_pr, p_wd = pctl_within(idx, pr), pctl_within(idx, wdeg)
        p_ty = pctl_within(idx, typicality)
        # relevance percentiles are ranked ONLY over papers that have a valid trajectory,
        # so current-year papers do not distort the cohort they are excluded from.
        idx_rel = [gi for gi in idx if not no_relevance[gi]]
        p_cy = pctl_within(idx_rel, cyr) if idx_rel else {}
        p_rw = pctl_within(idx_rel, raw) if idx_rel else {}
        rows = []
        for gi in idx:
            cent = (p_pr[gi] + p_wd[gi]) / 2
            rep = p_ty[gi]
            if no_relevance[gi]:
                # relevance axis dropped -> aggregate over the two remaining axes only
                score = aggregate_axes([cent, rep], [w[0], w[1]], mode=aggregate)
                rel = None
            else:
                rel_val = p_cy[gi] if relevance_mode == "cpy" else (p_cy[gi] + p_rw[gi]) / 2
                score = aggregate_axes([cent, rep, rel_val], [w[0], w[1], w[2]], mode=aggregate)
                rel = float(rel_val)
            rows.append(dict(index=gi, score=float(score),
                             centrality=float(cent), relevance=rel,
                             representativeness=float(rep), relevance_dropped=bool(no_relevance[gi]),
                             cited_by_count=int(raw[gi]),
                             cites_per_year=(None if no_relevance[gi] else float(cyr[gi]))))
        rows.sort(key=lambda d: d["score"], reverse=True)
        if split_current_year:
            # partition BEFORE truncation so each cohort gets its own top-n
            out_hist[cl] = [r for r in rows if not r["relevance_dropped"]][:n]
            out_front[cl] = [r for r in rows if r["relevance_dropped"]][:n]
        else:
            out[cl] = rows[:n]
    if split_current_year:
        return {"historical": out_hist, "fronts": out_front}
    return out


def render_cluster_block(lines, cl, rows, records, label_map, venue_of, kind, rel_desc):
    """Append one cluster's markdown table. kind in {'historical','fronts'}."""
    label = f" — {label_map[cl]}" if label_map and cl in label_map else ""
    lines.append(f"\n### C{cl}{label}\n")
    if kind == "fronts":
        lines.append("| # | Article | Annee | Revue | Score | Centralite | Typicite | Citations a ce jour |")
        lines.append("|---|---------|-------|-------|-------|------------|----------|---------------------|")
    else:
        lines.append("| # | Article | Annee | Revue | Score | Centralite | Pertinence | Typicite | Citations |")
        lines.append("|---|---------|-------|-------|-------|------------|------------|----------|-----------|")
    if not rows:
        ncol = 8 if kind == "fronts" else 9
        lines.append("| " + " | ".join(["—"] * ncol) + " |")
        return
    for rank, row in enumerate(rows, 1):
        r = records[row["index"]]
        doi = (r.get("doi") or "").replace("https://doi.org/", "")
        title = (r.get("title") or "").replace("|", "\\|")
        if doi and not doi.startswith("10.48550"):
            link = f"[{title}](https://doi.org/{doi})"
        else:
            link = title
        venue = venue_of(r) if venue_of else "—"
        seed = " · **graine**" if r.get("_is_seed") else ""
        if kind == "fronts":
            lines.append(f"| {rank} | {link}{seed} | {r.get('publication_year')} | {venue} | "
                         f"{row['score']:.2f} | {row['centrality']:.2f} | "
                         f"{row['representativeness']:.2f} | {row['cited_by_count']} |")
        else:
            rel_s = f"{row['relevance']:.2f}"
            cit_s = f"{row['cited_by_count']} ({row['cites_per_year']:.1f}/an)"
            lines.append(f"| {rank} | {link}{seed} | {r.get('publication_year')} | {venue} | "
                         f"{row['score']:.2f} | {row['centrality']:.2f} | {rel_s} | "
                         f"{row['representativeness']:.2f} | {cit_s} |")


def write_top_papers_3axis_md(top, records, path="papiers_pertinents_3axes.md",
                              label_map=None, venue_of=None,
                              relevance_mode="blend"):
    """Write the three-axis top-N selection to a markdown table file (non-split output)."""
    rel_desc = ("citations par an (moyenne) seules" if relevance_mode == "cpy"
                else "moyenne des percentiles de citations/an et de citations brutes")
    lines = ["# Articles les plus pertinents par cluster — indicateur composite 3 axes\n",
             "*Score composite à pondération egale (1/3 chacun), rangs percentiles intra-cluster :*\n",
             "1. **Centralite** = moyenne des percentiles de PageRank de couplage et du degre pondere (partage de references).",
             "2. **Pertinence** = " + rel_desc + ".",
             "3. **Representativite** = typicite thematique (cosinus au centroide TF-IDF du cluster).\n"]
    for cl in sorted(top):
        label = f" — {label_map[cl]}" if label_map and cl in label_map else ""
        lines.append(f"\n## C{cl}{label}\n")
        lines.append("| # | Article | Annee | Revue | Score | Centralite | Pertinence | Typicite | Citations |")
        lines.append("|---|---------|-------|-------|-------|------------|------------|----------|-----------|")
        for rank, row in enumerate(top[cl], 1):
            r = records[row["index"]]
            doi = (r.get("doi") or "").replace("https://doi.org/", "")
            title = (r.get("title") or "").replace("|", "\\|")
            if doi and not doi.startswith("10.48550"):
                link = f"[{title}](https://doi.org/{doi})"
            else:
                link = title
            venue = venue_of(r) if venue_of else "—"
            seed = " · **graine**" if r.get("_is_seed") else ""
            if row.get("relevance_dropped"):
                rel_s = "n.d. (annee en cours)"
                cit_s = f"{row['cited_by_count']}"
            else:
                rel_s = f"{row['relevance']:.2f}"
                cit_s = f"{row['cited_by_count']} ({row['cites_per_year']:.1f}/an)"
            lines.append(f"| {rank} | {link}{seed} | {r.get('publication_year')} | {venue} | "
                         f"{row['score']:.2f} | {row['centrality']:.2f} | {rel_s} | "
                         f"{row['representativeness']:.2f} | {cit_s} |")
    md = "\n".join(lines)
    with open(path, "w") as f:
        f.write(md)
    return path


def write_top_papers_split_md(split, records, path="papiers_pertinents_3axes_split.md",
                              label_map=None, venue_of=None, relevance_mode="blend",
                              current_year=None):
    """Write the SPLIT output (from top_papers_3axis(..., split_current_year=True)) to a
    two-part markdown file: established literature (3 axes) vs current-year research fronts
    (2 axes). `split` is the dict {"historical": {...}, "fronts": {...}}."""
    rel_desc = ("citations par an (moyenne) seules" if relevance_mode == "cpy"
                else "moyenne des percentiles de citations/an et de citations brutes")
    cy = f" ({current_year})" if current_year else ""
    lines = ["# Articles les plus pertinents par cluster — indicateur composite 3 axes\n",
             "Les articles de l'annee en cours sont separes : n'ayant pas d'annee complete "
             "de citations, ils ne sont pas comparables a la litterature etablie sur l'impact. "
             "Ils sont classes a part, sur ce qui est mesurable (centralite + typicite).\n",
             "\n## Partie 1 — Litterature etablie (annees completes)\n",
             "*Score composite a ponderation egale (1/3 chacun), rangs percentiles intra-cluster :*\n",
             "1. **Centralite** = moyenne des percentiles de PageRank de couplage et du degre pondere.",
             "2. **Pertinence** = " + rel_desc + ".",
             "3. **Representativite** = typicite thematique (cosinus au centroide TF-IDF du cluster).\n"]
    for cl in sorted(split["historical"]):
        render_cluster_block(lines, cl, split["historical"][cl], records,
                              label_map, venue_of, "historical", rel_desc)
    lines.append(f"\n## Partie 2 — Fronts de recherche{cy} — annee en cours\n")
    lines.append("*Articles de l'annee en cours, classes sur deux axes seulement "
                 "(la pertinence par citations n'est pas encore definie) :*\n")
    lines.append("1. **Centralite** = position dans le reseau de couplage (PageRank + degre pondere).")
    lines.append("2. **Typicite** = proximite thematique au centroide du cluster.")
    lines.append("*Le nombre de citations est indicatif (accumulation partielle).*\n")
    for cl in sorted(split["fronts"]):
        render_cluster_block(lines, cl, split["fronts"][cl], records,
                              label_map, venue_of, "fronts", rel_desc)
    md = "\n".join(lines)
    with open(path, "w") as f:
        f.write(md)
    return path
