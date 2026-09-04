"""
Topic sections: what each topic is, how each behaves, and how they relate.

Three sections, one subject — the BERTopic partition itself: the per-topic
description, the citation profile of each topic, and the similarity structure
between them.
"""

import numpy as np

from pysyrev.core.report.sections.common import (_safe_parse, _topic_cols,
                                                 _topic_label, _wo_outliers)


def _build_topics_section(topic_info, sections_cfg, topic_labels, nb_topics, section_n):
    """Section 1 — human-readable topic overview."""
    use_labels = topic_labels is not None
    nr_repr = sections_cfg.topics.n_repr_docs_per_topic

    # Summary table
    headers = ["ID", "LLM label" if use_labels else "Name", "# Docs", "Top words"]
    summary_rows = []
    for _, t in topic_info.iterrows():
        topic_id = int(t.get("Topic", -1))
        if topic_id == -1:
            continue
        name = _topic_label(topic_id, topic_labels)
        summary_rows.append([
            str(topic_id),
            name,
            str(int(t.get("Count", 0))),
            str(t.get("Representation", "-")),
        ])

    blocks = [
        {
            "type":       "table",
            "title":      f"{nb_topics} topics identified",
            "headers":    headers,
            "rows":       summary_rows,
            "col_widths": [1.0, 5.0, 1.5, 9.5],  # ID | Label/Name | #Docs | Top words — 17 cm
        }
    ]

    # Per-topic detail subsections with representative docs
    repr_meta = ["title", "year", "cited_by", "document_type", "doi"]
    for _, t in topic_info.iterrows():
        topic_id = int(t.get("Topic", -1))
        if topic_id == -1:
            continue

        label = _topic_label(topic_id, topic_labels)
        keywords = str(t.get("Representation", "-"))

        # Parse repr_doc columns
        repr_rows = []
        parsed = {
            col: _safe_parse(t.get(f"repr_doc_{col}", []))
            for col in repr_meta
            if f"repr_doc_{col}" in t.index
        }
        n_docs = min(nr_repr, len(parsed.get("title", [])))
        for i in range(n_docs):
            row = []
            for col in repr_meta:
                vals = parsed.get(col, [])
                val = vals[i] if i < len(vals) else "-"
                if col == "year" and val not in ("-", None):
                    try:
                        val = str(int(float(val)))
                    except (ValueError, TypeError):
                        val = str(val)
                elif col == "cited_by" and val not in ("-", None):
                    try:
                        val = str(int(float(val)))
                    except (ValueError, TypeError):
                        val = str(val)
                else:
                    val = "-" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
                row.append(val)
            repr_rows.append(row)

        sub_blocks = [
            {"type": "paragraph", "text": f"<b>Keywords:</b> {keywords}"},
        ]
        if repr_rows:
            sub_blocks.append({
                "type":    "table",
                "headers": ["Title", "Year", "Citations", "Type", "DOI"],
                "rows":    repr_rows,
                "col_widths": [7.5, 1.2, 1.5, 1.8, 5.0],  # Title | Year | Cit. | Type | DOI — 17 cm
            })

        blocks.append({
            "type":   "subsection",
            "title":  f"Topic {topic_id} — {label}",
            "blocks": sub_blocks,
        })

    return {"title": f"{section_n}. Topics", "blocks": blocks}


def _build_topic_characteristics_section(bertopic_results, topic_labels,
                                          chars_cfg, section_n):
    """Section — topic size and citation impact."""
    import plotly.express as px

    br_wo = _wo_outliers(bertopic_results)
    topic_ids = sorted(br_wo["Topic"].unique())
    labels    = [_topic_label(int(tid), topic_labels) for tid in topic_ids]

    sub_blocks = []

    # -- nb docs per topic --
    counts = [len(br_wo[br_wo["Topic"] == tid]) for tid in topic_ids]
    fig = px.bar(x=labels, y=counts,
                 labels={"x": "Topic", "y": "Nb documents"},
                 title="Number of documents per topic")
    fig.update_layout(width=850, height=450)
    sub_blocks.append({
        "type":   "subsection",
        "title":  "Number of documents per topic",
        "blocks": [{"type": "plotly", "figure": fig}],
    })

    # -- citation impact (only if cited_by is available) --
    if "cited_by" in br_wo.columns:
        total_cit = [
            br_wo.loc[br_wo["Topic"] == tid, "cited_by"].sum()
            for tid in topic_ids
        ]
        mean_cit  = [
            br_wo.loc[br_wo["Topic"] == tid, "cited_by"].mean()
            for tid in topic_ids
        ]

        fig_tot = px.bar(x=labels, y=total_cit,
                         labels={"x": "Topic", "y": "Total citations"},
                         title="Total citations per topic")
        fig_tot.update_layout(width=850, height=450)

        fig_avg = px.bar(x=labels, y=mean_cit,
                         labels={"x": "Topic", "y": "Mean citations"},
                         title="Mean citations per topic")
        fig_avg.update_layout(width=850, height=450)

        # Top M per topic: sum citations of the M most cited docs within each topic
        n_per = chars_cfg.n_top_cited_per_topic
        top_cit_per_topic = [
            br_wo.loc[br_wo["Topic"] == tid, "cited_by"]
            .nlargest(n_per).sum()
            for tid in topic_ids
        ]
        fig_top_per = px.bar(x=labels, y=top_cit_per_topic,
                             labels={"x": "Topic", "y": "Total citations"},
                             title=f"Total citations — top {n_per} most cited documents per topic")
        fig_top_per.update_layout(width=850, height=450)

        sub_blocks.append({
            "type":   "subsection",
            "title":  "Citation impact",
            "blocks": [
                {"type": "plotly", "figure": fig_tot,
                 "caption": "Total citation count per topic."},
                {"type": "plotly", "figure": fig_avg,
                 "caption": "Mean citation count per topic (normalised by topic size)."},
                {"type": "plotly", "figure": fig_top_per,
                 "caption": f"Sum of citations of the {n_per} most cited documents within each topic."},
            ],
        })

        # -- top N globally --
        n_global = chars_cfg.n_top_cited_global
        most_cited = (
            br_wo.sort_values("cited_by", ascending=False)
            .head(n_global)
        )
        topic_dist = (
            most_cited.groupby("Topic").size()
            .reset_index(name="count")
        )
        topic_dist["label"] = topic_dist["Topic"].apply(
            lambda tid: _topic_label(int(tid), topic_labels)
        )
        fig_glob = px.bar(
            topic_dist, x="label", y="count",
            labels={"label": "Topic", "count": "Count"},
            title=f"Topic distribution among the {n_global} most cited documents",
        )
        fig_glob.update_layout(width=850, height=450)
        sub_blocks.append({
            "type":   "subsection",
            "title":  f"Topics in the {n_global} most cited documents",
            "blocks": [{"type": "plotly", "figure": fig_glob}],
        })

    if not sub_blocks:
        return None
    return {"title": f"{section_n}. Topic characteristics", "blocks": sub_blocks}


def _build_topic_similarity_section(bertopic_results, topic_labels, sim_cfg, section_n):
    """Section — cosine similarity heatmap with optional hierarchical clustering."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim_fn
    from pysyrev.core.report.figures import plot_similarity_matrix
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    br_wo = _wo_outliers(bertopic_results)
    tcols = _topic_cols(br_wo)
    if not tcols:
        return None

    topic_ids = [int(c.split("#")[1]) for c in tcols]
    labels    = [_topic_label(tid, topic_labels) for tid in topic_ids]

    X = br_wo[tcols].values            # (n_docs, n_topics)
    cos_mat = cos_sim_fn(X.T)          # (n_topics, n_topics)

    sub_blocks = []
    order = list(range(len(topic_ids)))

    if sim_cfg.clustering:
        dist_sq = 1.0 - cos_mat.copy()
        np.fill_diagonal(dist_sq, 0.0)
        try:
            Z     = linkage(squareform(dist_sq), method="ward")
            order = list(leaves_list(Z))
        except Exception:
            pass  # fall back to natural order

        if sim_cfg.dendrogram:
            try:
                import plotly.figure_factory as ff
                from scipy.spatial.distance import pdist as scipy_pdist

                # ff.create_dendrogram passes distfun(X) → condensed distances
                # to linkagefun, not the raw data matrix.
                # We compute cosine distances explicitly and run ward linkage on them.
                reordered_labels = [labels[i] for i in order]
                fig_dend = ff.create_dendrogram(
                    X.T,
                    orientation="bottom",
                    labels=reordered_labels,
                    distfun=lambda data: scipy_pdist(data, metric="cosine"),
                    linkagefun=lambda d: linkage(d, method="ward"),
                )
                # Same tick angle, background and margins as the heatmap below,
                # so both figures of the section read as one pair.
                fig_dend.update_layout(
                    title=None,
                    width=900, height=380,
                    xaxis={"tickangle": -30, "automargin": True},
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="white",
                )
                sub_blocks.append({
                    "type": "plotly", "figure": fig_dend,
                    "caption": "Hierarchical clustering dendrogram (Ward, cosine distance). "
                               "It sets the row/column order of the heatmap below.",
                    "filename_prefix": "topic_similarity_dendrogram",
                })
            except Exception:
                pass

    # Reorder heatmap. The diagonal is blanked: a topic's similarity to itself
    # is 1 by construction and would flatten the colour scale of everything else.
    cos_plot = cos_mat.copy()
    np.fill_diagonal(cos_plot, np.nan)
    cos_plot       = cos_plot[np.ix_(order, order)]
    heat_labels    = [labels[i] for i in order]

    fig_heat = plot_similarity_matrix(cos_plot, heat_labels, title=None)
    sub_blocks.append({
        "type": "paragraph",
        "text": ("Pairwise cosine similarity between topics, computed from the "
                 "BERTopic probability distributions: how much two topics claim "
                 "the same documents. Unlike the connectivity panels, this says "
                 "nothing about shared references — two topics can be "
                 "semantically close yet bibliographically disconnected."),
    })
    sub_blocks.append({
        "type": "plotly", "figure": fig_heat,
        "caption": ("Darker = topics competing for the same documents. The "
                    "diagonal is blank (self-similarity is 1 by construction)."),
        "filename_prefix": "topic_similarity",
    })

    return {"title": f"{section_n}. Topic similarity", "blocks": sub_blocks}
