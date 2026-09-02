---
name: cluster-top-papers-3axis
description: Rank the most relevant papers of each cluster in a bibliometric corpus map using a THREE-AXIS composite indicator. Companion to corpus-map; an alternative to cluster-top-papers. Selects top-N per cluster by an equal-weighted composite of centrality (coupling PageRank + weighted degree), relevance (citations/year over complete years + raw citations), and representativeness (thematic typicality = cosine to the cluster TF-IDF centroid). Citations/year drops the incomplete current year from the denominator; current-year papers are scored on centrality + representativeness only. Use when representativeness should mean thematic prototypicality rather than coupling degree, or when absolute + age-normalised citation impact should both count. Supports mean (default), geometric-mean, or Chebyshev/maximin aggregation across the three axes via aggregate=.
---

# Cluster top papers — three-axis composite

Alternative to `cluster-top-papers`. Same equal-weighted, rank-percentile
philosophy, but the three axes are redefined and each combines two sub-metrics
(except representativeness):

## Axes

1. **Centrality** = mean of two within-cluster percentiles:
   - coupling **PageRank** (global prestige of a paper's couplings), and
   - weighted **degree** (raw strength of shared-reference overlap).
2. **Relevance** = mean of two within-cluster percentiles:
   - **citations per year** (age-normalised), and
   - **raw citations** (absolute impact).
   Pass `relevance_mode="cpy"` to use citations-per-year ALONE.
   Citations/year is computed over **complete years only**: with
   `drop_current_year=True` (default) the current (incomplete) year is removed
   from the denominator, so a paper published `y` years ago is divided by the
   number of full years elapsed, not by a partial current year.
3. **Representativeness** = **thematic typicality**: cosine similarity of the
   paper's TF-IDF vector to its cluster centroid — how prototypical its wording is.

Each sub-metric is converted to a within-cluster percentile (0-1, higher = better),
sub-metrics are averaged within an axis, and the three axes are averaged with equal
weight. Higher composite = better.

## Aggregation across axes

By default the three axis scores are combined by **arithmetic mean** — a *compensatory*
rule: a very strong axis can offset two weak ones. Pass `aggregate=` to change this:

- `"mean"` (default) — weighted arithmetic mean. Compensatory. A paper strong on one
  axis (e.g. highly central and cited but thematically atypical) can still rank high.
- `"gmean"` — weighted geometric mean. A weak axis pulls the score down more than under
  the mean, without zeroing it. A middle ground.
- `"chebyshev"` — augmented Tchebycheff / **maximin**: the composite is governed by the
  paper's **weakest** axis (a small mean term breaks ties). *Non-compensatory* — a paper
  ranks high only if it is strong on **every** axis at once. Use it for a top-N of
  all-round pivotal papers rather than single-axis specialists.

Note that the **median / majority-judgment** rule is deliberately NOT offered: with only
three axes the median is the middle axis, which *discards* both the best and the worst
axis and so rewards papers that collapse on one dimension (e.g. typicality ~0) — the
opposite of what the typicality axis is meant to enforce. `chebyshev` is the principled
non-compensatory alternative.

**Current-year papers are scored on two axes.** A paper published in the current
(incomplete) year has no complete year of citation accumulation, so its relevance is
**not computable**. Such papers are scored on **centrality + representativeness only**,
with the two weights renormalised (the relevance axis is dropped, not set to zero). In
the output their `relevance` is `None` and `relevance_dropped=True`; in the markdown
table their relevance cell reads "n.d. (annee en cours)". This lets a recent,
structurally central paper surface on what is measurable about it, without inventing a
citation impact it cannot yet have. The within-cluster relevance percentiles are ranked
over the valid-trajectory papers only, so current-year papers do not distort that cohort.

**Splitting established literature from research fronts.** Mixing current-year papers
into the same ranking with a blank relevance cell can be unsatisfying: the two cohorts
are not comparable on citation impact. Pass `split_current_year=True` to separate them.
The function then returns `{"historical": {cluster: [...]}, "fronts": {cluster: [...]}}`:
the **historical** table holds only papers with at least one complete citation year,
ranked on all three axes; the **fronts** table holds the current-year papers, ranked on
centrality + typicality alone. Each cohort is truncated to its own top-N, so a strong
established paper is no longer displaced from the top-N by a current-year paper sitting
at the rank boundary. Render it with `write_top_papers_split_md(...)`.

**No recency guardrail** (unlike cluster-top-papers): for papers with at least one
complete year, the relevance axis blends raw + per-year, so they are not double-penalised.
If you switch to `relevance_mode="cpy"`, recent high-rate papers can rank on
not-yet-stabilised counts.

## Kernel helpers (auto-loaded)

- `thematic_typicality(records, labels, text_of=None, ...)` -> array of cosine-to-centroid typicality.
- `top_papers_3axis(records, labels, W, n=3, weights=None, current_year=None, typicality=None, relevance_mode="blend", drop_current_year=True, ...)`
  -> dict `cluster -> [ {index, score, centrality, relevance, representativeness, cited_by_count, cites_per_year}, ... ]`.
- `write_top_papers_3axis_md(top, records, path=..., label_map=None, venue_of=None, relevance_mode="blend")` -> writes the markdown table.
- `top_papers_3axis(..., split_current_year=True)` -> `{"historical": {...}, "fronts": {...}}` instead of a flat per-cluster dict.
- `write_top_papers_split_md(split, records, path=..., label_map=None, venue_of=None, relevance_mode="blend", current_year=None)` -> writes the two-part markdown (established literature + current-year fronts).

## Workflow

Run after corpus-map clustering, with the clustered `corpus`, the integer `labels`
array (cluster per record, -1 = tail), and the symmetric bibliographic-coupling
matrix `W` in the kernel.

```python
top = top_papers_3axis(corpus, labels, W, n=3)            # blended relevance
write_top_papers_3axis_md(top, corpus, label_map=CLUSTER_LABELS, venue_of=venue_of)

# variant: citations-per-year alone on the relevance axis
top_cpy = top_papers_3axis(corpus, labels, W, n=3, relevance_mode="cpy")

# variant: non-compensatory aggregation (rank on the weakest axis)
top_ch = top_papers_3axis(corpus, labels, W, n=5, aggregate="chebyshev")

# variant: separate established literature from current-year research fronts
split = top_papers_3axis(corpus, labels, W, n=5, current_year=2026, split_current_year=True)
write_top_papers_split_md(split, corpus, label_map=CLUSTER_LABELS, venue_of=venue_of,
                          current_year=2026)
```

Save with `save_artifacts([...], language="python")`.

## Notes and variants

- **Different weighting.** `weights=(w_centrality, w_representativeness, w_relevance)`; auto-normalised.
- **Precomputed typicality.** Pass `typicality=<array>` to reuse a centroid-cosine vector.
- **Field keys.** Override `year_key` / `cite_key` if the schema differs.
- **Keep the current year in the denominator.** Pass `drop_current_year=False` to
  divide by `current_year - year + 1` (partial current year included) and score every
  paper on all three axes. Default `True` is recommended for a corpus that includes the
  running year.
- **Difference from `cluster-top-papers`.** That skill uses coupling **degree** for
  representativeness and a within-**year** citation percentile with a recency guardrail;
  this one uses **thematic typicality** for representativeness and a blended raw+per-year
  relevance axis. The two select overlapping but distinct top-N sets — this one favours
  thematically prototypical, historically well-cited papers.
