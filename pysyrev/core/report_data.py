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
# Network visualisation
# =============================================================================

def _make_network_figure(G, title: str, max_nodes: int = None,
                         topic_map: dict = None, topic_label_map: dict = None,
                         directed: bool = False):
    """Return a Plotly Figure representing a NetworkX graph.

    When *topic_map* ({node_id: topic_int}) is provided nodes are coloured by
    topic with a discrete palette and a legend; otherwise they are coloured by
    degree.  Large graphs are subsampled to the top *max_nodes* nodes by degree
    before layout computation.

    For directed graphs (*directed=True*) nodes are sized by in-degree and
    arrowheads are drawn on edges when the graph has ≤ 300 edges.

    Returns (figure, is_truncated).
    """
    import plotly.graph_objects as go
    import plotly.colors
    from collections import defaultdict as _dd

    is_truncated = False
    if max_nodes and G.number_of_nodes() > max_nodes:
        top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()
        is_truncated = True

    import networkx as nx
    pos = nx.spring_layout(G, seed=42)

    # For directed graphs, size by in-degree (how often a node is cited);
    # for undirected graphs, use total degree.
    if directed:
        degrees = dict(G.in_degree())
    else:
        degrees = dict(G.degree())

    nodes   = list(G.nodes())
    max_deg = max((degrees[n] for n in nodes), default=1)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#CCCCCC"),
        hoverinfo="none",
        showlegend=False,
    )

    # Arrow annotations for directed graphs (skipped when too many edges)
    arrow_annotations = []
    if directed and G.number_of_edges() <= 300:
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            arrow_annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowwidth=1, arrowcolor="#AAAAAA",
            ))

    # Per-node hover text
    def _hover(n):
        attrs = G.nodes[n]
        parts = [f"<b>{_node_display_label(n, attrs)}</b>"]
        for attr_key, lbl in [("year", "Year"), ("journal", "Journal"), ("doi", "DOI")]:
            val = attrs.get(attr_key)
            if val and str(val) not in ("-", "nan", "None"):
                parts.append(f"{lbl}: {str(val)[:60]}")
        parts.append(f"Degree: {degrees[n]}")
        if topic_map is not None:
            t = topic_map.get(n)
            if t is not None and t != -1:
                parts.append(f"Topic: {(topic_label_map or {}).get(t, f'Topic {t}')}")
        return "<br>".join(parts)

    hover_texts = [_hover(n) for n in nodes]
    node_size   = [6 + 14 * (degrees[n] / max_deg) for n in nodes]
    palette     = plotly.colors.qualitative.D3  # 10 distinct colours

    # Node traces: one per topic (discrete legend) or single degree-coloured trace
    if topic_map:
        groups = _dd(list)
        for i, n in enumerate(nodes):
            groups[topic_map.get(n)].append(i)

        node_traces = []
        for t in sorted(groups, key=lambda x: (x is None, x == -1, x or 0)):
            idxs = groups[t]
            if t is None or t == -1:
                color = "#BBBBBB"
                name  = "Outlier / no topic"
            else:
                color = palette[t % len(palette)]
                name  = (topic_label_map or {}).get(t, f"Topic {t}")
            node_traces.append(go.Scatter(
                x=[pos[nodes[i]][0] for i in idxs],
                y=[pos[nodes[i]][1] for i in idxs],
                mode="markers",
                name=name,
                hovertext=[hover_texts[i] for i in idxs],
                hoverinfo="text",
                marker=dict(
                    size=[node_size[i] for i in idxs],
                    color=color,
                    line=dict(width=0.5, color="white"),
                ),
            ))
    else:
        node_traces = [go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers",
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            marker=dict(
                size=node_size,
                color=[degrees[n] for n in nodes],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Degree", thickness=12),
                line=dict(width=0.5, color="white"),
            ),
        )]

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=title if not is_truncated else f"{title} (top {max_nodes} nodes by degree)",
            showlegend=topic_map is not None,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
            width=900, height=600,
            annotations=arrow_annotations if arrow_annotations else [],
        ),
    )
    return fig, is_truncated


# =============================================================================
# Section builders
# =============================================================================

def _build_overview_section(run_dir, model_index, best_results, section_n):
    row = best_results.iloc[model_index]
    coherence_col = next(
        (c for c in best_results.columns if c not in _FIXED_METRIC_COLS), None
    )
    kv_items = [
        {"key": "Run directory",         "value": run_dir},
        {"key": "Selected model (rank)", "value": model_index},
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


def _node_display_label(node_id: str, attrs: dict) -> str:
    """Human-readable label for a graph node.

    Priority: title attribute → label attribute → node_id stripped of its
    R:/U: prefix (set by build_cocitation_graph).
    """
    if attrs.get("title"):
        return str(attrs["title"])
    if attrs.get("label"):
        return str(attrs["label"])
    s = str(node_id)
    return s[2:] if len(s) > 2 and s[1] == ":" else s


def _build_bib_network_section(bib_network_config, section_n,
                               bertopic_results=None, topic_labels=None,
                               export_to=None, section_cfg=None):
    """Section — bibliographic coupling and co-citation networks."""
    try:
        import networkx as nx
    except ImportError:
        return None

    # Co-citation graphs are typically large; cap visualisation at this many nodes.
    _COCITATION_MAX_NODES = 150

    # (path, label, max_nodes, node_col_header, is_directed)
    graphs = [
        (bib_network_config.citation_graph,   "Citation",               None,                  "Title",     True),
        (bib_network_config.coupling_graph,   "Bibliographic coupling", None,                  "Title",     False),
        (bib_network_config.cocitation_graph, "Co-citation",            _COCITATION_MAX_NODES, "Reference", False),
    ]

    # Build a base topic map {paper_id: topic_int} and outlier set from bertopic_results
    _id_topic_base: dict = {}
    _outlier_ids: set = set()
    if bertopic_results is not None and "Topic" in bertopic_results.columns:
        id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
        if id_col:
            _id_topic_base = dict(zip(bertopic_results[id_col], bertopic_results["Topic"]))
            _outlier_ids = set(
                bertopic_results.loc[bertopic_results["Topic"] == -1, id_col]
            )

    _corpus_only  = section_cfg.corpus_only if section_cfg is not None else True
    _excl_outliers = section_cfg.exclude_outliers      if section_cfg is not None else True

    sub_blocks = []
    for path, label, max_nodes, node_col_header, is_directed in graphs:
        if not path or not os.path.isfile(path):
            continue
        G = nx.read_graphml(path)

        is_cocitation = (not is_directed) and (max_nodes is not None)

        # ── Filter nodes ──────────────────────────────────────────────────────
        if is_cocitation:
            # Keep only nodes that belong to the corpus (have a title attribute)
            if _corpus_only:
                G = G.subgraph([n for n in G.nodes() if G.nodes[n].get("title")]).copy()
            # Remove outlier corpus nodes (Topic == -1)
            if _excl_outliers and _outlier_ids:
                G = G.subgraph(
                    [n for n in G.nodes() if n[2:] not in _outlier_ids]
                ).copy()
        elif is_directed:
            # Citation: remove external nodes when corpus_only, then outliers
            if _corpus_only:
                G = G.subgraph(
                    [n for n in G.nodes() if G.nodes[n].get("node_type") != "external"]
                ).copy()
            if _excl_outliers and _outlier_ids:
                G = G.subgraph(
                    [n for n in G.nodes() if n not in _outlier_ids]
                ).copy()
        else:
            # Coupling: nodes are corpus doc IDs — just remove outliers
            if _excl_outliers and _outlier_ids:
                G = G.subgraph(
                    [n for n in G.nodes() if n not in _outlier_ids]
                ).copy()

        # Remove isolated nodes that became disconnected after filtering
        G = G.subgraph([n for n, d in G.degree() if d > 0]).copy()

        n_nodes  = G.number_of_nodes()
        n_edges  = G.number_of_edges()
        if n_nodes == 0:
            continue
        density  = nx.density(G)
        avg_deg  = (n_edges / n_nodes) if (is_directed and n_nodes) else (
                   (2 * n_edges / n_nodes) if n_nodes else 0.0)

        # topic_map keyed by graph node IDs
        if _id_topic_base:
            if is_cocitation:
                # Co-citation nodes are prefixed with R: or U:
                topic_map = {f"R:{pid}": t for pid, t in _id_topic_base.items()}
            else:
                topic_map = dict(_id_topic_base)
        else:
            topic_map = None

        # Top 10 by in-degree (citation) or total degree (coupling/co-citation)
        if is_directed:
            degree_seq = sorted(dict(G.in_degree()).items(), key=lambda x: x[1], reverse=True)
            deg_header = "In-degree"
        else:
            degree_seq = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)
            deg_header = "Degree"

        top_nodes = degree_seq[:10]
        node_rows = []
        for node_id, deg in top_nodes:
            attrs    = G.nodes[node_id]
            display  = _node_display_label(node_id, attrs)
            node_doi = attrs.get("doi", "-")
            row = [display, str(deg), str(node_doi)[:50]]
            if is_cocitation:
                row.append("Yes" if attrs.get("title") else "No")
            node_rows.append(row)

        avg_deg_label = "Average in-degree" if is_directed else "Average degree"
        sub_content = [
            {
                "type": "key_value",
                "items": [
                    {"key": "Nodes",          "value": n_nodes},
                    {"key": "Edges",          "value": n_edges},
                    {"key": "Density",        "value": f"{density:.4f}"},
                    {"key": avg_deg_label,    "value": f"{avg_deg:.2f}"},
                ],
            }
        ]
        if node_rows:
            if is_cocitation:
                headers    = [node_col_header, deg_header, "DOI", "In corpus"]
                col_widths = [8.5, 1.5, 4.5, 2.5]
            else:
                headers    = [node_col_header, deg_header, "DOI"]
                col_widths = [10.0, 2.0, 5.0]
            sub_content.append({
                "type":       "table",
                "title":      "Top 10 most connected nodes",
                "headers":    headers,
                "rows":       node_rows,
                "col_widths": col_widths,
            })

        # Network figure — coloured by topic when available
        fig, is_truncated = _make_network_figure(
            G, label,
            max_nodes=max_nodes,
            topic_map=topic_map,
            topic_label_map=topic_labels,
            directed=is_directed,
        )
        if is_directed:
            caption = "Node size scales with in-degree (number of times cited by other corpus documents)."
        else:
            caption = "Node size scales with degree."
        if topic_map:
            caption += " Node colour indicates topic assignment."
        if is_truncated:
            caption += f" Only the {max_nodes} highest-degree nodes are shown."
        sub_content.append({
            "type":            "plotly",
            "figure":          fig,
            "caption":         caption,
            "filename_prefix": label.lower().replace(" ", "_"),
        })

        # Export interactive HTML alongside the PDF
        if export_to:
            os.makedirs(export_to, exist_ok=True)
            html_name = label.lower().replace(" ", "_") + "_network.html"
            html_path = os.path.join(export_to, html_name)
            fig.write_html(html_path)
            file_uri  = Path(html_path).resolve().as_uri()
            sub_content.append({
                "type":  "callout",
                "title": "Interactive version",
                "text":  (
                    f'Open the interactive graph: '
                    f'<link href="{file_uri}" color="#0B63CE">{html_name}</link>'
                ),
            })

        sub_blocks.append({
            "type":   "subsection",
            "title":  label,
            "blocks": sub_content,
        })

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
            title="Number of documents per topic over time (absolute)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Absolute",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Total documents per topic per year."}],
        })

    # -- cumulative --
    if "cumulative" in variants:
        fig = px.histogram(
            br_wo.sort_values("Topic"),
            x="year", y="nb_docs", color="Topic",
            title="Number of documents per topic over time (cumulative)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            cumulative=True,
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Cumulative",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Cumulative documents per topic per year."}],
        })

    # -- normalized --
    if "normalized" in variants:
        topic_sizes = br_wo.groupby("Topic")["nb_docs"].transform("sum")
        br_norm = br_wo.copy()
        br_norm["normalized"] = 1.0 / topic_sizes
        fig = px.histogram(
            br_norm.sort_values("Topic"),
            x="year", y="normalized", color="Topic",
            title="Proportion of each topic's documents over time (normalized)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Normalized (within-topic proportion)",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Each topic's share of its own documents per year."}],
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
                title="Weighted number of documents per topic over time",
                nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            )
            fig.update_layout(width=900, height=550)
            sub_blocks.append({
                "type":   "subsection",
                "title":  "Weighted (BERTopic approximate distribution)",
                "blocks": [{"type": "plotly", "figure": fig,
                            "caption": "Sum of BERTopic topic-probability weights per year."}],
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


def _build_paper_selection_section(bertopic_results, topic_info, topic_labels,
                                   sel_cfg, export_to, section_n,
                                   coupling_graph=None, cocitation_graph=None):
    """Section — paper reading list (synthetic table + annex export)."""
    REQUIRED = {"year", "cited_by", "document_type", "title", "doi", "Topic"}
    if not REQUIRED.issubset(bertopic_results.columns):
        return None

    br_wo = _wo_outliers(bertopic_results).copy()
    br_wo["year"]     = pd.to_numeric(br_wo["year"],     errors="coerce")
    br_wo["cited_by"] = pd.to_numeric(br_wo["cited_by"], errors="coerce").fillna(0)

    # Precompute degree map for network-based selection modes.
    # Coupling nodes are doc IDs directly; co-citation nodes carry a R: prefix.
    _degree_map: dict = {}
    if sel_cfg.selection_by == "coupling" and coupling_graph is not None:
        _degree_map = dict(coupling_graph.degree())
        selection_label = "Most central (coupling)"
    elif sel_cfg.selection_by == "co_citation" and cocitation_graph is not None:
        _degree_map = {n[2:]: d for n, d in cocitation_graph.degree()
                       if n.startswith("R:")}
        selection_label = "Most central (co-citation)"
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
        elif sel_cfg.selection_by in ("coupling", "co_citation") and _degree_map:
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

    table_rows = []
    for _, row in full_df.iterrows():
        tid   = int(row["Topic"])
        label = _topic_label(tid, topic_labels)
        table_rows.append([
            str(tid),
            label,
            str(row.get("title", "-"))[:100],
            str(row.get("document_type", "-")),
            _fmt_year(row.get("year")),
            _fmt_cit(row.get("cited_by")),
            str(row.get("doi", "-")),
            str(row.get("_selection", "-")),
        ])

    # Export annex
    annex_msg = None
    if sel_cfg.export_annex and export_to:
        os.makedirs(export_to, exist_ok=True)
        ext      = sel_cfg.annex_format.lower()
        ann_path = os.path.join(export_to, f"paper_selection.{ext}")
        if ext == "csv":
            full_df.drop(columns=["_selection"], errors="ignore").rename(
                columns={"_selection": "selection_type"}
            )
            export_df = full_df.rename(columns={"_selection": "selection_type"})
            export_df.to_csv(ann_path, index=False)
        else:
            with open(ann_path, "w", encoding="utf-8") as f:
                headers_txt = ["topic", "label", "title", "type", "year", "citations", "doi", "selection"]
                f.write(" | ".join(headers_txt) + "\n")
                f.write("-" * 120 + "\n")
                for r in table_rows:
                    f.write(" | ".join(r) + "\n")
        annex_msg = f"Full selection ({len(full_df)} papers) exported to: {ann_path}"

    blocks = [
        {
            "type":       "table",
            "title":      f"Paper reading list — {len(full_df)} papers selected",
            "headers":    ["Topic", "Label", "Title", "Type", "Year", "Cit.", "DOI", "Selection"],
            "rows":       table_rows,
            "col_widths": [1.0, 2.5, 5.5, 1.5, 1.0, 1.0, 2.5, 2.0],  # Topic|Label|Title|Type|Year|Cit.|DOI|Sel. — 17 cm
        }
    ]
    if annex_msg:
        blocks.append({"type": "paragraph", "text": annex_msg})

    return {"title": f"{section_n}. Paper selection", "blocks": blocks}


# =============================================================================
# Main orchestrator
# =============================================================================

def build_report_data(run_dir: str,
                      model_index: int,
                      best_results: pd.DataFrame,
                      topic_info: pd.DataFrame,
                      bertopic_results: pd.DataFrame,
                      report_config,
                      bib_network_config=None,
                      topic_labels: dict = None,
                      export_to: str = None) -> dict:
    """Build the declarative report_data dict consumed by PDFReportEngine."""
    meta = report_config.meta
    sec  = report_config.sections
    nb_topics = int(best_results.iloc[model_index]["nb_topics"])

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

    # Load bib-network graphs once for reuse in network-based paper selection
    _coupling_graph   = None
    _cocitation_graph = None
    if bib_network_config is not None:
        try:
            import networkx as _nx
            if getattr(bib_network_config, 'coupling_graph', None) and os.path.isfile(bib_network_config.coupling_graph):
                _coupling_graph = _nx.read_graphml(bib_network_config.coupling_graph)
            if getattr(bib_network_config, 'cocitation_graph', None) and os.path.isfile(bib_network_config.cocitation_graph):
                _cocitation_graph = _nx.read_graphml(bib_network_config.cocitation_graph)
        except Exception:
            pass

    # 2. Bib networks (conditional)
    bib_enabled = str(sec.bib_network.enabled).lower()
    if bib_enabled in ("true", "auto") and bib_network_config is not None:
        section = _build_bib_network_section(
            bib_network_config, n,
            bertopic_results=bertopic_results,
            topic_labels=topic_labels,
            export_to=export_to,
            section_cfg=sec.bib_network,
        )
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
        coupling_graph=_coupling_graph,
        cocitation_graph=_cocitation_graph,
    )
    if section is not None:
        report_data["sections"].append(section)
        n += 1

    # 7. Technical appendix
    report_data["sections"].append(
        _build_overview_section(run_dir, model_index, best_results, n)
    )

    # Extra user-defined sections
    if sec.extra:
        for extra_section in sec.extra:
            report_data["sections"].append(extra_section)

    return report_data
