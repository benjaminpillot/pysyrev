"""
The reading list: which papers of each topic are worth reading first.

Ranked by citations, by coupling centrality, or by the three-axis composite
that asks all three questions at once — is this paper central to its
sub-literature, typical of its topic, and actually cited?
"""

import os

import numpy as np
import pandas as pd

from pysyrev.core.report.sections.common import _topic_label, _wo_outliers


def _composite_scores(coupling_result, bertopic_results, *, aggregate="mean",
                      weights=None, relevance_mode="blend",
                      drop_current_year=True, current_year=None):
    """Three-axis ranking of papers within their Leiden coupling community,
    keyed by document id.

    Returns ``{doc_id: {score, centrality, relevance, representativeness,
    relevance_dropped}}`` (each axis a within-community percentile). Faithful to
    the network-first paradigm: papers are ranked inside their bibliographic-
    coupling community (``coupling_result.labels``), not inside the global
    BERTopic topic. Combines coupling centrality (PageRank + weighted degree on
    the coupling matrix), citation relevance (per-year + raw) and thematic
    representativeness (typicality to the *community*'s text centroid). The
    ``aggregate``, ``weights``, ``relevance_mode``, ``drop_current_year`` and
    ``current_year`` knobs are forwarded to
    :func:`pysyrev.core.paper_ranking.top_papers_3axis`. Empty dict when a
    prerequisite is missing.
    """
    if coupling_result is None or coupling_result.n_nodes < 2:
        return {}
    if bertopic_results is None:
        return {}
    id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
    if id_col is None:
        return {}

    by_id = bertopic_results.drop_duplicates(id_col).set_index(id_col)
    text_col = "Document" if "Document" in by_id.columns else "title"

    def _num(v, default=0.0):
        try:
            return default if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return default

    node_ids = coupling_result.node_ids
    comm = np.asarray(coupling_result.labels)   # Leiden community per node (-1 = tail)
    records, labels = [], []
    for i, nid in enumerate(node_ids):
        if nid in by_id.index:
            row = by_id.loc[nid]
            year = row.get("year")
            records.append({
                "publication_year": (None if pd.isna(year) else int(_num(year))),
                "cited_by_count": _num(row.get("cited_by")),
                "_text": str(row.get(text_col) or ""),
            })
        else:
            records.append({"publication_year": None, "cited_by_count": 0.0, "_text": ""})
        labels.append(int(comm[i]))

    from pysyrev.core.paper_ranking import top_papers_3axis
    top = top_papers_3axis(records, labels, coupling_result.W, n=len(records),
                           weights=(tuple(weights) if weights else None),
                           current_year=current_year, relevance_mode=relevance_mode,
                           drop_current_year=drop_current_year, aggregate=aggregate,
                           text_of=lambda r: r.get("_text", ""))
    detail = {}
    for rows in top.values():
        for r in rows:
            detail[node_ids[r["index"]]] = {
                "score": r["score"],
                "centrality": r["centrality"],
                "relevance": r["relevance"],
                "representativeness": r["representativeness"],
                "relevance_dropped": r["relevance_dropped"],
            }
    return detail


def _is_review(df, detect):
    """Boolean mask of the literature reviews in *df*, per the ``detect`` mode.

    ``document_type`` is a controlled vocabulary, so it is matched component-wise
    over its ``;``-separated parts — "Review", "Review; Early Access", OpenAlex's
    lowercase "review" — rather than as a substring. The title test is a plain
    substring, since a title has no vocabulary to lean on.

    Peer review defeats both tests and is filtered out of both. OpenAlex's
    ``peer-review`` type is a referee report, not a literature review, and its
    titles read "Reviewer #2 (Public Review): ..." or "Author response: ..."; that
    type is excluded outright, whatever the mode. A title naming peer review is
    dropped too, since those are papers *about* refereeing rather than reviews of
    a literature — at the price of missing an actual review of peer-review practice.

    Both modes earn their keep: OpenAlex types reviews far more sparingly than WoS,
    so `"type"` alone misses the genuine reviews an OpenAlex-heavy corpus declares
    only in its title.
    """
    dtype = df["document_type"].fillna("").astype(str).str.lower()
    title = df["title"].fillna("").astype(str).str.lower()

    parts     = [[p.strip() for p in s.split(";")] for s in dtype]
    referee   = pd.Series([any(p in ("peer-review", "peer review") for p in ps)
                           for ps in parts], index=df.index, dtype=bool)
    type_hit  = pd.Series([any(p == "review" for p in ps) for ps in parts],
                          index=df.index, dtype=bool)
    title_hit = (title.str.contains("review")
                 & ~title.str.contains("peer[- ]review", regex=True))

    hit = title_hit if detect == "title" else (
        type_hit if detect == "type" else type_hit | title_hit)
    return hit & ~referee


def _review_relevance(reviews, relevance_mode, current_year):
    """Citation relevance of each review, as a percentile within *reviews* —
    same vocabulary as the composite's relevance axis. Unlike the composite the
    current year stays in the cit/yr denominator: relevance is this branch's only
    axis, so it cannot be dropped for a current-year review.
    """
    from pysyrev.core.paper_ranking import pctl_within

    raw = reviews["cited_by"].astype(float).fillna(0.0).to_numpy()
    year = reviews["year"].astype(float).to_numpy()
    elapsed = np.maximum(current_year - np.nan_to_num(year, nan=current_year) + 1, 1.0)
    cpy = raw / elapsed

    idx = list(range(len(reviews)))
    p_cy = pctl_within(idx, cpy)
    if relevance_mode == "cpy":
        vals = [p_cy[i] for i in idx]
    else:
        p_rw = pctl_within(idx, raw)
        vals = [(p_cy[i] + p_rw[i]) / 2 for i in idx]
    return pd.Series(vals, index=reviews.index, dtype=float)


def _pick_reviews(reviews, cfg, current_year):
    """A topic's reviews, as a recent cohort followed by a ranked historical one.

    A review published within ``recent_years`` of the corpus's last year has had
    no time to accrue citations, so ranking it on citation relevance would bury
    it under any mediocre older review — and a review of the current year is
    exactly the one worth reading. It bypasses that ranking and keeps a
    guaranteed slot instead, capped on its own by ``max_recent_reviews`` and
    ordered newest first. Only the rest goes through ``_review_relevance``.
    """
    is_recent = reviews["year"].fillna(-np.inf) > current_year - cfg.recent_years
    recent = (reviews[is_recent]
              .sort_values(["year", "cited_by"], ascending=False)
              .head(cfg.max_recent_reviews))

    hist = reviews[~is_recent]
    if len(hist):
        rel = _review_relevance(hist, cfg.relevance_mode, current_year)
        hist = (hist.assign(_rel=rel)
                    .sort_values("_rel", ascending=False)
                    .head(cfg.max_reviews_per_topic)
                    .drop(columns=["_rel"]))
    return pd.concat([recent, hist])


def _build_paper_selection_section(bertopic_results, topic_info, topic_labels,
                                   sel_cfg, export_to, section_n,
                                   coupling_result=None, cocitation_result=None):
    """Section — paper reading list (synthetic table + annex export)."""
    REQUIRED = {"year", "cited_by", "document_type", "title", "doi", "Topic"}
    if not REQUIRED.issubset(bertopic_results.columns):
        return None

    br_wo = _wo_outliers(bertopic_results).copy()
    br_wo["year"]     = pd.to_numeric(br_wo["year"],     errors="coerce")
    br_wo["cited_by"] = pd.to_numeric(br_wo["cited_by"], errors="coerce").fillna(0)

    # Precompute a {doc_id: strength} map for network-based selection modes.
    # Coupling nodes are document IDs; co-citation nodes are reference IDs — a
    # corpus document is ranked by its own co-citation strength when it is
    # itself cited (its id appears as a reference).
    def _strength_map(result):
        return {nid: float(s) for nid, s in
                zip(result.node_ids, result.W.sum(axis=1))} if result is not None else {}

    _degree_map: dict = {}
    _composite_detail: dict = {}
    if sel_cfg.selection_by == "coupling" and coupling_result is not None:
        _degree_map = _strength_map(coupling_result)
        selection_label = "Most central (coupling)"
    elif sel_cfg.selection_by == "co_citation" and cocitation_result is not None:
        _degree_map = _strength_map(cocitation_result)
        selection_label = "Most central (co-citation)"
    elif sel_cfg.selection_by == "composite" and coupling_result is not None:
        comp = sel_cfg.composite
        _composite_detail = _composite_scores(
            coupling_result, bertopic_results,
            aggregate=comp.aggregate, weights=comp.weights,
            relevance_mode=comp.relevance_mode,
            drop_current_year=comp.drop_current_year,
            current_year=comp.current_year,
        )
        _degree_map = {nid: d["score"] for nid, d in _composite_detail.items()}
        selection_label = "Most relevant (3-axis)"
    else:
        selection_label = ""

    # Composite split: current-year papers (relevance axis dropped) are pulled out
    # of the historical 3-axis list into a separate "research fronts" table, so a
    # 2-axis paper is never ranked against 3-axis ones.
    id_col = next((c for c in ["id", "ID"] if c in br_wo.columns), None)
    comp_split = (sel_cfg.selection_by == "composite" and bool(_composite_detail)
                  and sel_cfg.composite.split_current_year and id_col is not None)
    front_ids = ({i for i, d in _composite_detail.items() if d.get("relevance_dropped")}
                 if comp_split else set())

    # Review branch: reference year for the citations/year denominator. The
    # corpus's latest year, so the branch stands alone whatever `selection_by` is.
    rev_cfg  = sel_cfg.review
    rev_year = float(br_wo["year"].max()) if br_wo["year"].notna().any() else 0.0

    selected_parts = []
    for topic_id in sorted(br_wo["Topic"].unique()):
        topic_df = br_wo[br_wo["Topic"] == topic_id].copy()
        filt     = topic_df[topic_df["year"] >= sel_cfg.min_year]
        if front_ids:                       # historical list = non-front papers only
            filt = filt[~filt[id_col].isin(front_ids)]

        # Reviews — an opt-out branch that runs beside (not through) the ranked
        # selection, so it is ranked and capped on its own terms.
        if rev_cfg.enabled:
            reviews = filt[_is_review(filt, rev_cfg.detect)].copy()
            if len(reviews):
                reviews = _pick_reviews(reviews, rev_cfg, rev_year)
        else:
            reviews = filt.iloc[0:0].copy()
        reviews["_selection"] = "Review"

        n_docs = max(1, round(sel_cfg.proportion_per_topic * len(topic_df)))
        if sel_cfg.selection_by == "random":
            selected = filt.sample(n=min(n_docs, len(filt)), random_state=42).copy()
            selected["_selection"] = "Random"
            selection_label = "Random"
        elif sel_cfg.selection_by in ("coupling", "co_citation", "composite") and _degree_map:
            id_col = next((c for c in ["id", "ID"] if c in filt.columns), None)
            if id_col:
                tmp = filt.copy()
                tmp["_degree"] = tmp[id_col].map(_degree_map).fillna(0)
                selected = tmp.sort_values("_degree", ascending=False).head(n_docs).copy()
                selected = selected.drop(columns=["_degree"])
            else:
                selected = filt.sort_values("cited_by", ascending=False).head(n_docs).copy()
            selected["_selection"] = selection_label
        else:  # "citations" (default, or fallback when graph is unavailable)
            selected = filt.sort_values("cited_by", ascending=False).head(n_docs).copy()
            selected["_selection"] = "Most cited"
            selection_label = "Most cited"

        combined = pd.concat([reviews, selected]).drop_duplicates(subset=["doi"])

        # Mark papers in both categories
        rev_dois = set(reviews["doi"].dropna())
        sel_dois = set(selected["doi"].dropna())
        both     = rev_dois & sel_dois
        if both:
            combined.loc[combined["doi"].isin(both), "_selection"] = f"Review + {selection_label}"

        selected_parts.append(combined)

    if not selected_parts:
        return None

    full_df = pd.concat(selected_parts, ignore_index=True)

    # Composite mode: attach the three axis scores as columns (for the table and
    # the CSV annex), replacing Type/DOI with Score/Centrality/Relevance/Typicality.
    id_col = next((c for c in ["id", "ID"] if c in full_df.columns), None)
    is_composite = (sel_cfg.selection_by == "composite"
                    and bool(_composite_detail) and id_col is not None)
    if is_composite:
        det = _composite_detail
        for col, key in [("composite_score", "score"), ("centrality", "centrality"),
                         ("relevance", "relevance"), ("representativeness", "representativeness"),
                         ("_reldrop", "relevance_dropped")]:
            full_df[col] = full_df[id_col].map(lambda i: (det.get(i) or {}).get(key))

    # Research fronts frame (current-year cohort), built once and reused for both
    # the fronts table and the CSV annex. It is a per-topic SELECTION — the same
    # top-N-per-topic proportion as the historical list — so the two cohorts are
    # comparably sized, not the whole current year. Tag a `cohort` column so the
    # annex holds both in one file, distinguishable by that column.
    fronts_export = None
    if comp_split and front_ids:
        det = _composite_detail
        fronts_all = br_wo[br_wo[id_col].isin(front_ids)].copy()
        for col, key in [("composite_score", "score"), ("centrality", "centrality"),
                         ("relevance", "relevance"),
                         ("representativeness", "representativeness"),
                         ("_reldrop", "relevance_dropped")]:
            fronts_all[col] = fronts_all[id_col].map(lambda i: (det.get(i) or {}).get(key))
        parts = []
        for topic_id in sorted(fronts_all["Topic"].unique()):
            tdf = fronts_all[fronts_all["Topic"] == topic_id].sort_values(
                "composite_score", ascending=False)
            n_f = max(1, round(sel_cfg.proportion_per_topic * len(tdf)))
            parts.append(tdf.head(n_f))
        fronts_export = (pd.concat(parts, ignore_index=True) if parts
                         else fronts_all.iloc[0:0])
        fronts_export["_selection"] = "Research front"
        full_df["cohort"] = "historical"
        fronts_export["cohort"] = "front"

    def _fmt_year(v):
        try:
            return str(int(float(v)))
        except Exception:
            return "-"

    def _fmt_cit(v):
        try:
            return str(int(float(v)))
        except Exception:
            return "-"

    def _fmt_axis(v):
        try:
            return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}"
        except Exception:
            return "-"

    if is_composite:
        headers    = ["Topic", "Label", "Title", "Year", "Cit.",
                      "Score", "Cent.", "Rel.", "Typ.", "Selection"]
        col_widths = [1.0, 2.2, 5.2, 0.9, 0.9, 1.1, 1.1, 1.1, 1.1, 2.4]  # 17 cm
    else:
        headers    = ["Topic", "Label", "Title", "Type", "Year", "Cit.", "DOI", "Selection"]
        col_widths = [1.0, 2.5, 5.5, 1.5, 1.0, 1.0, 2.5, 2.0]  # 17 cm

    table_rows = []
    for _, row in full_df.iterrows():
        tid   = int(row["Topic"])
        label = _topic_label(tid, topic_labels)
        cells = [str(tid), label, str(row.get("title", "-"))[:100]]
        if is_composite:
            rel = "n.d." if row.get("_reldrop") else _fmt_axis(row.get("relevance"))
            cells += [
                _fmt_year(row.get("year")), _fmt_cit(row.get("cited_by")),
                _fmt_axis(row.get("composite_score")), _fmt_axis(row.get("centrality")),
                rel, _fmt_axis(row.get("representativeness")),
                str(row.get("_selection", "-")),
            ]
        else:
            cells += [
                str(row.get("document_type", "-")), _fmt_year(row.get("year")),
                _fmt_cit(row.get("cited_by")), str(row.get("doi", "-")),
                str(row.get("_selection", "-")),
            ]
        table_rows.append(cells)

    # Export annex
    annex_msg = None
    if sel_cfg.export_annex and export_to:
        os.makedirs(export_to, exist_ok=True)
        ext      = sel_cfg.annex_format.lower()
        ann_path = os.path.join(export_to, f"paper_selection.{ext}")
        # Historical + fronts in one file, distinguished by the `cohort` column.
        src = (pd.concat([full_df, fronts_export], ignore_index=True)
               if fronts_export is not None else full_df)
        if ext == "csv":
            export_df = src.drop(columns=["_reldrop"], errors="ignore").rename(
                columns={"_selection": "selection_type",
                         "representativeness": "typicality"})
            export_df.to_csv(ann_path, index=False)
        else:
            with open(ann_path, "w", encoding="utf-8") as f:
                f.write(" | ".join(h.lower() for h in headers) + "\n")
                f.write("-" * 120 + "\n")
                for r in table_rows:
                    f.write(" | ".join(r) + "\n")
        annex_msg = f"Full selection ({len(src)} papers) exported to: {ann_path}"

    blocks = [
        {
            "type":       "table",
            "title":      f"Paper reading list — {len(full_df)} papers selected",
            "headers":    headers,
            "rows":       table_rows,
            "col_widths": col_widths,
        }
    ]
    if annex_msg:
        blocks.append({"type": "paragraph", "text": annex_msg})

    # Research fronts: the current-year selection (already top-N per topic in
    # fronts_export), ranked on 2 axes, shown separately.
    if fronts_export is not None:
        front_rows = []
        ordered = fronts_export.sort_values(
            ["Topic", "composite_score"], ascending=[True, False])
        for _, row in ordered.iterrows():
            tid = int(row["Topic"])
            front_rows.append([
                str(tid), _topic_label(tid, topic_labels),
                str(row.get("title", "-"))[:100],
                _fmt_year(row.get("year")), _fmt_cit(row.get("cited_by")),
                _fmt_axis(row.get("composite_score")),
                _fmt_axis(row.get("centrality")),
                _fmt_axis(row.get("representativeness")),
            ])
        if front_rows:
            blocks.append({
                "type":       "table",
                "title":      f"Research fronts — current-year papers ({len(front_rows)})",
                "headers":    ["Topic", "Label", "Title", "Year", "Cit.",
                               "Score", "Cent.", "Typ."],
                "rows":       front_rows,
                "col_widths": [1.0, 2.6, 6.0, 0.9, 0.9, 1.2, 1.2, 1.2],
            })
            blocks.append({"type": "paragraph", "text": (
                "Research fronts are the current (incomplete) year's papers, ranked on "
                "centrality + typicality only — the citation-relevance axis needs a "
                "complete year and is not yet meaningful for them. They are kept out of "
                "the historical 3-axis list above so a 2-axis score is never compared "
                "against a 3-axis one.")})

    return {"title": f"{section_n}. Paper selection", "blocks": blocks}
