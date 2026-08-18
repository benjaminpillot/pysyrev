"""
Data-loading and report-building helpers for the topic-report pipeline.

Called by TopicReport (topic_report.py). Core contains pure functions;
the runtime class coordinates them and holds instance-level cache.
"""

import ast
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


_FIXED_METRIC_COLS = frozenset({
    "hdbscan", "umap", "nb_topics", "entropy", "diversity",
    "nb_outliers", "distance", "model", "purity", "distance_with_purity",
})


# =============================================================================
# File helpers
# =============================================================================

def find_best_results_csv(run_dir: str) -> Tuple[str, str]:
    """Locate the unique *_best_results.csv under run_dir/metrics/.

    Returns (csv_path, distance_name).
    """
    pattern = os.path.join(run_dir, "metrics", "*_best_results.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No best_results CSV found under {run_dir}/metrics/. "
            "Run TopicModel first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple best_results CSVs found under {run_dir}/metrics/: "
            f"{matches}. Expected exactly one."
        )
    path = matches[0]
    distance_name = Path(path).stem.replace("_best_results", "")
    return path, distance_name


def build_file_prefix(hdbscan_params: str, umap_params: str,
                      distance_name: str) -> str:
    return f"hdbscan={hdbscan_params}_umap={umap_params}_distance={distance_name}"


def load_topic_info(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "topic_info", f"{file_prefix}.csv")
    return pd.read_csv(path)


def load_bertopic_results(run_dir: str, file_prefix: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "bertopic_results", f"{file_prefix}.csv")
    return pd.read_csv(path)


# =============================================================================
# Helpers
# =============================================================================

def _safe_parse(val):
    """Parse a stringified Python list stored in a CSV cell."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _topic_label(topic_id: int, topic_labels: Optional[dict]) -> str:
    if topic_labels and topic_id in topic_labels:
        return topic_labels[topic_id]
    return f"Topic {topic_id}"


def _wo_outliers(bertopic_results: pd.DataFrame) -> pd.DataFrame:
    return bertopic_results[bertopic_results["Topic"] != -1].copy()


def _topic_cols(bertopic_results: pd.DataFrame):
    return sorted(
        [c for c in bertopic_results.columns if c.startswith("topic#")],
        key=lambda c: int(c.split("#")[1])
    )


# =============================================================================
# Section builders
# =============================================================================

def _build_overview_section(run_dir, best_model_index, best_results, section_n):
    row = best_results.iloc[best_model_index]
    coherence_col = next(
        (c for c in best_results.columns if c not in _FIXED_METRIC_COLS), None
    )
    kv_items = [
        {"key": "Run directory",         "value": run_dir},
        {"key": "Selected model (rank)", "value": best_model_index},
        {"key": "HDBSCAN parameters",    "value": row["hdbscan"]},
        {"key": "UMAP parameters",       "value": row["umap"]},
        {"key": "Number of topics",      "value": int(row["nb_topics"])},
        {"key": "Number of outliers",    "value": int(row["nb_outliers"])},
        {"key": "Entropy",               "value": f"{row['entropy']:.4f}"},
        {"key": "Diversity",             "value": f"{row['diversity']:.4f}"},
        {"key": "Purity",                "value": f"{row['purity']:.4f}"},
    ]
    if coherence_col:
        kv_items.insert(6, {
            "key":   f"Coherence ({coherence_col})",
            "value": f"{row[coherence_col]:.4f}",
        })
    return {
        "title": f"{section_n}. Topic model — Technical overview",
        "blocks": [{"type": "key_value", "items": kv_items}],
    }


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


def _export_network_html(fig, export_to, label):
    """Write an interactive HTML version of a Plotly network figure and return a
    'Interactive version' callout block (or None)."""
    if not export_to:
        return None
    import plotly.graph_objects as _go
    os.makedirs(export_to, exist_ok=True)
    html_name = label.lower().replace(" ", "_") + "_network.html"
    html_path = os.path.join(export_to, html_name)
    fig_html = _go.Figure(fig)
    fig_html.update_layout(autosize=True, width=None, height=None)
    fig_html.write_html(
        html_path,
        config={"responsive": True},
        default_width="100%", default_height="100%",
        post_script=(
            "document.documentElement.style.cssText='height:100%;margin:0;padding:0;';"
            "document.body.style.cssText='height:100%;margin:0;padding:0;';"
        ),
    )
    file_uri = Path(html_path).resolve().as_uri()
    return {
        "type":  "callout",
        "title": "Interactive version",
        "text":  (f'Open the interactive graph: '
                  f'<link href="{file_uri}" color="#0B63CE">{html_name}</link>'),
    }


def _build_network_subsection(result, df, bertopic_results, topic_labels, *,
                              title, filename_prefix, node_kind, count_label,
                              k, color_mode, hulls, export_to):
    """Render one computed network (coupling or co-citation) as a subsection.

    *result* is a :class:`NetworkResult`; *df* supplies node metadata
    (titles/years) for nodes whose id is a corpus document. When *color_mode* is
    ``"topic"`` nodes are coloured by BERTopic topic (the croisement) — fully
    meaningful for document nodes; for reference nodes only in-corpus references
    carry a topic. Otherwise nodes are coloured by Leiden community.
    Returns a subsection block, or None when the network is empty.
    """
    from pysyrev.core.networks import plot_network, backbone_edges

    if result is None or result.n_nodes == 0:
        return None

    # Topic per node: document ids map directly; reference ids only when the
    # reference is itself a corpus document (otherwise -1 / grey).
    topic_of = {}
    if bertopic_results is not None and "Topic" in bertopic_results.columns:
        id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
        if id_col:
            topic_of = dict(zip(bertopic_results[id_col], bertopic_results["Topic"]))
    topics = np.array([int(topic_of.get(nid, -1)) for nid in result.node_ids])
    color_labels = {t: _topic_label(t, topic_labels)
                    for t in set(topics.tolist()) if t != -1}

    meta = df.drop_duplicates("id").set_index("id") if df is not None else None

    def _meta(nid, col, default="-"):
        if meta is not None and nid in meta.index:
            v = meta.at[nid, col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return default

    strength = result.W.sum(axis=1)
    hover = []
    for nid, comm, tp, st in zip(result.node_ids, result.labels, topics, strength):
        title_txt = str(_meta(nid, "title", nid))[:70]
        tlabel = color_labels.get(int(tp), "—")
        hover.append(
            f"<b>{title_txt}</b><br>Year: {_meta(nid, 'year')}<br>"
            f"Community: {'C%d' % comm if comm >= 0 else 'tail'}<br>"
            f"Topic: {tlabel}<br>Strength: {st:.3g}"
        )

    color_by = topics if color_mode == "topic" else None
    fig = plot_network(
        result.coords, result.labels, result.W, k=k,
        color_by=color_by, color_labels=color_labels,
        community_hulls=(hulls and color_mode == "topic"),
        hover_text=hover, title=None,
    )

    # ── Stats + top-10 nodes by strength ───────────────────────────────────
    n_backbone = len(backbone_edges(result.W, result.labels, k=k))
    n_full     = int((result.W > 0).sum() // 2)
    connected  = int((strength > 0).sum())
    sub_content = [{
        "type": "key_value",
        "items": [
            {"key": count_label,               "value": result.n_nodes},
            {"key": f"Connected {node_kind}s", "value": f"{connected} ({connected / result.n_nodes:.0%})"},
            {"key": "Leiden communities",      "value": result.n_communities},
            {"key": "Modularity",              "value": f"{result.modularity:.3f}"},
            {"key": "Edges",                   "value": n_full},
            {"key": f"Backbone edges (k={k})", "value": n_backbone},
        ],
    }]

    order = np.argsort(strength)[::-1][:10]
    rows = []
    for i in order:
        nid = result.node_ids[i]
        comm = int(result.labels[i])
        rows.append([
            str(_meta(nid, "title", nid))[:70],
            f"{strength[i]:.3g}",
            f"C{comm}" if comm >= 0 else "-",
            color_labels.get(int(topics[i]), "-"),
        ])
    if rows:
        sub_content.append({
            "type": "table",
            "title": f"Top 10 {node_kind}s by strength",
            "headers": [node_kind.capitalize(), "Strength", "Community", "Topic"],
            "rows": rows,
            "col_widths": [8.5, 1.8, 2.2, 4.5],
        })

    if color_mode == "topic":
        caption = ("Layout and blobs come from the network structure; node colour "
                   "is the BERTopic topic, so each community's topic mix is visible. "
                   "Node size scales with strength; grey nodes are unconnected.")
    else:
        caption = ("Layout, blobs and colour all come from the network structure "
                   "(Leiden communities). Node size scales with strength.")
    sub_content.append({
        "type": "plotly", "figure": fig,
        "caption": caption, "filename_prefix": filename_prefix,
    })

    html_block = _export_network_html(fig, export_to, title)
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection", "title": title, "blocks": sub_content}


def _build_connectivity_subsection(coupling_result, bertopic_results,
                                   topic_labels, section_cfg, export_to):
    """Inter-group connectivity: mean bibliographic coupling between topics (or
    Leiden communities), from the coupling matrix. A heatmap plus a per-group
    internal/outward table. Grouping by topic is the informative use — it tests
    whether topics share references (are intellectually connected) or are siloed.
    Returns a subsection block, or None when fewer than two groups exist.
    """
    from pysyrev.core.networks import (
        inter_cluster_matrix, corpus_baseline, coupling_inout,
        groups_from_labels, plot_connectivity_matrix,
    )

    if coupling_result is None or coupling_result.n_nodes < 2:
        return None

    by    = section_cfg.connectivity.by
    scale = section_cfg.connectivity.scale
    W     = coupling_result.W

    if by == "community":
        groups = groups_from_labels(coupling_result.labels)
        unit   = "community"
    else:
        topic_of = {}
        if bertopic_results is not None and "Topic" in bertopic_results.columns:
            id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
            if id_col:
                topic_of = dict(zip(bertopic_results[id_col], bertopic_results["Topic"]))
        topics = np.array([int(topic_of.get(nid, -1)) for nid in coupling_result.node_ids])
        groups = [(_topic_label(t, topic_labels), np.where(topics == t)[0])
                  for t in sorted(x for x in set(topics.tolist()) if x != -1)]
        unit = "topic"

    groups = [(lab, idx) for lab, idx in groups if len(idx) > 0]
    if len(groups) < 2:
        return None

    M, labs = inter_cluster_matrix(W, groups, scale=scale)
    base    = corpus_baseline(W, scale=scale)
    fig     = plot_connectivity_matrix(M, labs, baseline=base, title=None)

    rows = []
    for i, lab in enumerate(labs):
        p = coupling_inout(M, i)
        siloed = p["internal"] > p["outward"] and p["outward"] < base
        rows.append([lab, f"{p['internal']:.2f}", f"{p['outward']:.2f}",
                     "self-contained" if siloed else "integrated"])

    sub_content = [
        {"type": "paragraph",
         "text": (f"Mean bibliographic coupling between {unit}s (shared references, "
                  f"scaled ×{int(scale)}). The diagonal is each {unit}'s internal "
                  f"cohesion; off-diagonal cells show how much two {unit}s share "
                  f"references. Corpus baseline: <b>{base:.2f}</b>.")},
        {"type": "plotly", "figure": fig,
         "caption": (f"Darker = stronger shared-reference overlap. A {unit} with high "
                     f"internal and low outward coupling (below baseline) is a "
                     f"self-contained / weakly integrated theme."),
         "filename_prefix": f"{unit}_connectivity"},
        {"type": "table",
         "title": f"Internal vs outward coupling per {unit}",
         "headers": [unit.capitalize(), "Internal", "Outward", "Profile"],
         "rows": rows, "col_widths": [7.0, 3.0, 3.0, 4.0]},
    ]
    html_block = _export_network_html(fig, export_to, f"{unit} connectivity")
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection",
            "title": f"{unit.capitalize()} connectivity",
            "blocks": sub_content}


def _build_networks_section(df, coupling_result, cocitation_result,
                            bertopic_results, topic_labels, section_cfg,
                            export_to, section_n):
    """Section — bibliographic coupling and co-citation networks.

    Coupling and co-citation are recomputed from the reviewed dataset's raw
    references (Salton / co-citation matrix → Leiden communities → backbone
    layout) and rendered with Plotly; coupling is coloured by BERTopic topic (the
    croisement). A third panel gives the inter-topic connectivity matrix (mean
    coupling between topics). Citation is intentionally not rendered yet.
    """
    cfg = section_cfg
    sub_blocks = []

    coupling_sub = _build_network_subsection(
        coupling_result, df, bertopic_results, topic_labels,
        title="Bibliographic coupling", filename_prefix="bibliographic_coupling",
        node_kind="document", count_label="Documents",
        k=cfg.coupling.backbone_k, color_mode=cfg.coupling.color_by,
        hulls=cfg.coupling.hulls, export_to=export_to,
    )
    if coupling_sub is not None:
        sub_blocks.append(coupling_sub)

    cocitation_sub = _build_network_subsection(
        cocitation_result, df, bertopic_results, topic_labels,
        title="Co-citation", filename_prefix="co_citation",
        node_kind="reference", count_label="References",
        k=cfg.cocitation.backbone_k, color_mode=cfg.cocitation.color_by,
        hulls=cfg.cocitation.hulls, export_to=export_to,
    )
    if cocitation_sub is not None:
        sub_blocks.append(cocitation_sub)

    connectivity_sub = _build_connectivity_subsection(
        coupling_result, bertopic_results, topic_labels, cfg, export_to)
    if connectivity_sub is not None:
        sub_blocks.append(connectivity_sub)

    if not sub_blocks:
        return None
    return {"title": f"{section_n}. Bibliographic networks", "blocks": sub_blocks}


def _build_temporal_section(bertopic_results, nb_topics, temporal_cfg, section_n):
    """Section — number of documents per topic over time."""
    import plotly.express as px

    if "year" not in bertopic_results.columns:
        return None

    br_wo = _wo_outliers(bertopic_results).copy()
    br_wo = br_wo.dropna(subset=["year"])
    br_wo["year"]    = br_wo["year"].astype(int)
    br_wo["nb_docs"] = 1

    variants = temporal_cfg.variants
    sub_blocks = []

    # -- absolute --
    if "absolute" in variants:
        fig = px.histogram(
            br_wo.sort_values("Topic"),
            x="year", y="nb_docs", color="Topic",
            title="Number of documents per topic over time (cumulative absolute)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            cumulative=True,
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Absolute",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Cumulative total documents per topic per year."}],
        })

    # -- normalized --
    if "normalized" in variants:
        topic_sizes = br_wo.groupby("Topic")["nb_docs"].transform("sum")
        br_norm = br_wo.copy()
        br_norm["normalized"] = 1.0 / topic_sizes
        fig = px.histogram(
            br_norm.sort_values("Topic"),
            x="year", y="normalized", color="Topic",
            title="Proportion of each topic's documents over time (cumulative normalized)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            cumulative=True,
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Normalized (within-topic proportion)",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Cumulative share of each topic's own documents per year."}],
        })

    # -- weighted (BERTopic approximate distribution) --
    if "weighted" in variants:
        tcols = _topic_cols(br_wo)
        if tcols:
            id_year = br_wo[["year"]].copy()
            df_long = pd.concat([id_year, br_wo[tcols]], axis=1)
            df_long = df_long.melt(id_vars=["year"], var_name="topic", value_name="weight")
            df_long["topic_num"] = df_long["topic"].str.extract(r"(\d+)").astype(int)
            df_agg = (
                df_long
                .groupby(["year", "topic_num"], as_index=False)["weight"]
                .sum()
                .rename(columns={"weight": "weighted_count"})
            )
            fig = px.histogram(
                df_agg.sort_values("topic_num"),
                x="year", y="weighted_count", color="topic_num",
                title="Weighted number of documents per topic over time (cumulative)",
                nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
                cumulative=True,
            )
            fig.update_layout(width=900, height=550)
            sub_blocks.append({
                "type":   "subsection",
                "title":  "Weighted (BERTopic approximate distribution)",
                "blocks": [{"type": "plotly", "figure": fig,
                            "caption": "Cumulative sum of BERTopic topic-probability weights per year."}],
            })

    if not sub_blocks:
        return None
    return {"title": f"{section_n}. Temporal dynamics", "blocks": sub_blocks}


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
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim_fn
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
                fig_dend.update_layout(
                    title="Topic similarity — Dendrogram",
                    width=900, height=380,
                    xaxis={"tickangle": -45},
                )
                sub_blocks.append({
                    "type": "plotly", "figure": fig_dend,
                    "caption": "Hierarchical clustering dendrogram (Ward, cosine distance).",
                })
            except Exception:
                pass

    # Reorder heatmap
    cos_plot = cos_mat.copy()
    np.fill_diagonal(cos_plot, np.nan)
    cos_plot       = cos_plot[np.ix_(order, order)]
    heat_labels    = [labels[i] for i in order]

    fig_heat = go.Figure(go.Heatmap(
        z=cos_plot,
        x=heat_labels,
        y=heat_labels,
        colorscale="Viridis",
        colorbar=dict(title="Cosine"),
    ))
    fig_heat.update_layout(
        title="Cosine similarity between topics",
        width=850, height=750,
        xaxis={"tickangle": -45},
    )
    sub_blocks.append({
        "type": "plotly", "figure": fig_heat,
        "caption": "Pairwise cosine similarity computed from BERTopic probability distributions.",
    })

    return {"title": f"{section_n}. Topic similarity", "blocks": sub_blocks}


def _composite_scores(coupling_result, bertopic_results, *, aggregate="mean",
                      weights=None, relevance_mode="blend",
                      drop_current_year=True, current_year=None):
    """Three-axis ranking of papers within their topic, keyed by document id.

    Returns ``{doc_id: {score, centrality, relevance, representativeness,
    relevance_dropped}}`` (each axis a within-topic percentile). Combines
    coupling centrality (PageRank + weighted degree on the coupling matrix),
    citation relevance (per-year + raw) and thematic representativeness
    (typicality to the topic's TF-IDF centroid). The ``aggregate``, ``weights``,
    ``relevance_mode``, ``drop_current_year`` and ``current_year`` knobs are
    forwarded to :func:`pysyrev.core.paper_ranking.top_papers_3axis`. Empty dict
    when a prerequisite is missing.
    """
    if coupling_result is None or coupling_result.n_nodes < 2:
        return {}
    if bertopic_results is None or "Topic" not in bertopic_results.columns:
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
    records, labels = [], []
    for nid in node_ids:
        if nid in by_id.index:
            row = by_id.loc[nid]
            year = row.get("year")
            records.append({
                "publication_year": (None if pd.isna(year) else int(_num(year))),
                "cited_by_count": _num(row.get("cited_by")),
                "_text": str(row.get(text_col) or ""),
            })
            labels.append(int(_num(row.get("Topic"), -1)))
        else:
            records.append({"publication_year": None, "cited_by_count": 0.0, "_text": ""})
            labels.append(-1)

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

    selected_parts = []
    for topic_id in sorted(br_wo["Topic"].unique()):
        topic_df = br_wo[br_wo["Topic"] == topic_id].copy()
        filt     = topic_df[topic_df["year"] >= sel_cfg.min_year]

        # Reviews
        is_review = (
            filt["document_type"].str.lower().fillna("").str.contains("review")
            | filt["title"].str.lower().fillna("").str.contains("review")
        )
        reviews = filt[is_review].copy()
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
        if ext == "csv":
            export_df = full_df.drop(columns=["_reldrop"], errors="ignore").rename(
                columns={"_selection": "selection_type",
                         "representativeness": "typicality"})
            export_df.to_csv(ann_path, index=False)
        else:
            with open(ann_path, "w", encoding="utf-8") as f:
                f.write(" | ".join(h.lower() for h in headers) + "\n")
                f.write("-" * 120 + "\n")
                for r in table_rows:
                    f.write(" | ".join(r) + "\n")
        annex_msg = f"Full selection ({len(full_df)} papers) exported to: {ann_path}"

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

    return {"title": f"{section_n}. Paper selection", "blocks": blocks}


# =============================================================================
# Main orchestrator
# =============================================================================

def build_report_data(run_dir: str,
                      best_model_index: int,
                      best_results: pd.DataFrame,
                      topic_info: pd.DataFrame,
                      bertopic_results: pd.DataFrame,
                      report_config,
                      topic_labels: dict = None,
                      export_to: str = None,
                      coupling_dataset: str = None) -> dict:
    """Build the declarative report_data dict consumed by PDFReportEngine."""
    meta = report_config.meta
    sec  = report_config.sections
    nb_topics = int(best_results.iloc[best_model_index]["nb_topics"])

    report_data = {
        "meta": {
            "title":    meta.title,
            "subtitle": meta.subtitle,
            "author":   meta.author,
            "date":     datetime.now().strftime(meta.date_format),
            "version":  meta.version,
            "summary":  meta.summary,
        },
        "sections": [],
    }

    n = 1  # running section counter

    # 1. Topics
    report_data["sections"].append(
        _build_topics_section(topic_info, sec, topic_labels, nb_topics, n)
    )
    n += 1

    # Compute the coupling and co-citation networks once from the reviewed
    # dataset's raw references — shared by the networks section and by
    # network-based paper selection.
    _networks_df = _coupling_result = _cocitation_result = None
    if coupling_dataset:
        try:
            _networks_df = pd.read_csv(coupling_dataset, low_memory=False)
        except Exception:
            _networks_df = None
        if _networks_df is not None and {"references", "id"}.issubset(_networks_df.columns):
            from pysyrev.core.networks import build_coupling, build_cocitation
            nc = sec.bib_network
            try:
                _coupling_result = build_coupling(
                    _networks_df,
                    resolution=nc.coupling.resolution, min_size=nc.coupling.min_size)
            except Exception:
                _coupling_result = None
            try:
                _cocitation_result = build_cocitation(
                    _networks_df, min_ref_freq=nc.cocitation.min_ref_freq,
                    resolution=nc.cocitation.resolution, min_size=nc.cocitation.min_size)
            except Exception:
                _cocitation_result = None

    # 2. Bibliographic networks (coupling + co-citation) — rendered whenever the
    # reviewed dataset yielded a coupling network (i.e. it carried references).
    if _coupling_result is not None:
        section = _build_networks_section(
            _networks_df, _coupling_result, _cocitation_result,
            bertopic_results, topic_labels, sec.bib_network, export_to, n)
        if section is not None:
            report_data["sections"].append(section)
            n += 1

    # 3. Temporal dynamics
    section = _build_temporal_section(bertopic_results, nb_topics, sec.temporal, n)
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 4. Topic characteristics
    section = _build_topic_characteristics_section(
        bertopic_results, topic_labels, sec.topic_characteristics, n
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 5. Topic similarity
    section = _build_topic_similarity_section(
        bertopic_results, topic_labels, sec.topic_similarity, n
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 6. Paper selection
    section = _build_paper_selection_section(
        bertopic_results, topic_info, topic_labels, sec.paper_selection, export_to, n,
        coupling_result=_coupling_result,
        cocitation_result=_cocitation_result,
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 7. Technical appendix
    report_data["sections"].append(
        _build_overview_section(run_dir, best_model_index, best_results, n)
    )

    # Extra user-defined sections
    if sec.extra:
        for extra_section in sec.extra:
            report_data["sections"].append(extra_section)

    return report_data
