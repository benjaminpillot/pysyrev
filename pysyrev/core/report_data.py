"""
Data-loading and report-building helpers for the topic-report pipeline.

Called by TopicReport (topic_report.py). Core contains pure functions;
the runtime class coordinates them and holds instance-level cache.
"""

import ast
import glob
import os
import re
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


def _slug(label: str) -> str:
    """Filename-safe slug of a panel title.

    Panel titles carry punctuation that has no business in a filename — the ``×``
    of "Community × topic mapping", the parentheses of "Community connectivity
    (co-citation)". Collapse every run of non-alphanumerics into one underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _export_network_html(fig, export_to, label):
    """Write an interactive HTML version of a Plotly network figure and return a
    'Interactive version' callout block (or None)."""
    if not export_to:
        return None
    import plotly.graph_objects as _go
    os.makedirs(export_to, exist_ok=True)
    html_name = _slug(label) + "_network.html"
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


def _community_repr_titles(result, df, n=5) -> dict:
    """Top-strength paper titles per Leiden coupling community.

    Anchors the LLM community label with a few concrete papers. Returns
    ``{community_id: [titles]}``; empty when titles aren't available.
    """
    if result is None or df is None or "title" not in df.columns:
        return {}
    id_to_title = (df.drop_duplicates("id").set_index("id")["title"].to_dict()
                   if "id" in df.columns else {})
    strength = result.W.sum(axis=1)
    out: dict = {}
    for comm in set(int(c) for c in result.labels if c >= 0):
        idx = [i for i in range(len(result.labels))
               if int(result.labels[i]) == comm]
        idx.sort(key=lambda i: strength[i], reverse=True)
        titles = []
        for i in idx[:n]:
            t = id_to_title.get(result.node_ids[i])
            if isinstance(t, str) and t.strip():
                titles.append(t.strip())
        if titles:
            out[comm] = titles
    return out


def _metadata_fetchers(cfg) -> dict:
    """Build the ``{id_scheme: ids -> {id: metadata}}`` fetchers for
    :func:`~pysyrev.core.references.fetch_reference_metadata` from *cfg*.

    One provider registers the schemes it can serve, from its own credentials.
    OpenAlex is the only one implemented today — it answers on both its own ids
    and DOIs, so it covers both schemes; another metadata source would register
    here the same way. An unknown provider yields ``{}``, leaving the references
    unresolved rather than failing the report.
    """
    from functools import partial

    if cfg.provider == "openalex":
        from pysyrev.core.api.openalex_client import OpenAlexClient
        from pysyrev.core.references import fetch_metadata_via_openalex
        client = OpenAlexClient(api_key=cfg.api_key, email=cfg.email)
        fetch = partial(fetch_metadata_via_openalex, client=client,
                        include_abstracts=cfg.include_abstracts)
        return {"openalex": fetch, "doi": fetch}

    print(f"[report] unknown reference-metadata provider {cfg.provider!r}; "
          f"co-citation nodes left unresolved.")
    return {}


def _cocitation_metadata(df, ref_col, cocitation_cfg) -> dict:
    """Resolve the co-citation nodes to real works, per the panel's config.

    Returns ``{reference_key: metadata}``, or ``{}`` when the step is not
    configured. The node set is derived up front from the reference sets
    (:func:`~pysyrev.core.networks.frequent_references`) so only the references
    that will actually become nodes are looked up — the once-cited long tail,
    which ``min_ref_freq`` drops anyway, is never queried.

    Like every other enrichment here, a failure degrades rather than propagates:
    the co-citation panel still renders, just with unnamed clusters.
    """
    cfg = getattr(cocitation_cfg, "metadata", None)
    if cfg is None:
        return {}
    try:
        from pysyrev.core.networks import frequent_references, reference_sets
        from pysyrev.core.references import fetch_reference_metadata

        fetchers = _metadata_fetchers(cfg)
        if not fetchers:
            return {}
        _, refsets = reference_sets(df, ref_col=ref_col)
        keys, _ = frequent_references(refsets, min_ref_freq=cocitation_cfg.min_ref_freq)
        if not keys:
            return {}
        meta = fetch_reference_metadata(keys, fetchers, cache_path=cfg.cache)
        resolved = sum(1 for m in meta.values() if m.get("title"))
        print(f"[report] co-citation node metadata: {resolved}/{len(keys)} "
              f"references resolved via {cfg.provider}.")
        return meta
    except Exception as e:
        print(f"[report] co-citation node metadata skipped "
              f"(network still rendered): {e!r}")
        return {}


def _cocitation_repr_titles(result, n=5) -> dict:
    """Top-strength reference titles per Leiden co-citation community.

    The co-citation counterpart of :func:`_community_repr_titles`: titles come
    from the resolved node metadata rather than the corpus, since the references
    are mostly extra-corpus. Empty when the metadata step did not run.
    """
    meta = getattr(result, "node_meta", None)
    if result is None or not meta:
        return {}
    from pysyrev.core.networks import cluster_profiles
    profiles = cluster_profiles(result.node_ids, result.labels, result.W, meta,
                                top_works=n)
    out = {}
    for comm, profile in profiles.items():
        titles = [w["title"] for w in profile["works"] if w.get("title")]
        if titles:
            out[comm] = titles
    return out


def _label_communities(run_dir, result, config, repr_docs, kind):
    """Human-readable labels for one network's Leiden communities, cached on disk.

    Returns ``{community_id: label}`` or None. The LLM runs only on a cache miss
    (the cache is keyed on a hash of the community sub-theme terms), so an
    unchanged partition never triggers a call — the point of doing this once,
    not on every report. Coupling and co-citation therefore share the cache
    without colliding: different terms, different key.
    """
    if config is None or result is None:
        return None
    terms = getattr(result, "terms", None)
    if not terms:
        return None
    from pysyrev.core.topic_labels import (
        load_cached_cluster_labels, save_cluster_labels)

    cached = load_cached_cluster_labels(run_dir, terms)
    if cached is not None:
        return cached
    # The LLM label is optional enrichment: it must never break the networks
    # section or the whole report. Any failure (API error, rate limit, bad
    # response) degrades to no labels — the section still renders with the raw
    # TF-IDF terms. On success the labels are cached so the call happens once.
    try:
        from pysyrev.core.llm import label_clusters
        print(f"Generating human-readable {kind}-community labels via LLM…")
        labels = label_clusters(terms, config, repr_docs=repr_docs)
        path = save_cluster_labels(run_dir, terms, labels)
        print(f"{kind.capitalize()}-community labels saved to: {path}")
        return labels
    except Exception as e:
        print(f"[report] community labels skipped (network still rendered): {e!r}")
        return None


def _label_coupling_communities(run_dir, coupling_result, networks_df, config):
    """Labels for the coupling communities, anchored on their papers' titles."""
    return _label_communities(
        run_dir, coupling_result, config,
        repr_docs=_community_repr_titles(coupling_result, networks_df),
        kind="coupling")


def _label_cocitation_communities(run_dir, cocitation_result, config):
    """Labels for the co-citation communities, anchored on the most co-cited
    references' titles. Only possible once the node metadata is resolved."""
    return _label_communities(
        run_dir, cocitation_result, config,
        repr_docs=_cocitation_repr_titles(cocitation_result),
        kind="co-citation")


def _community_profile_blocks(result, node_meta) -> list:
    """Who and when each community is made of — for a network of references.

    Once the co-citation nodes are resolved to real works, a community can be
    read as an intellectual base: the authors it keeps citing, the venues that
    carry it, and the period it spans. Returns ``[]`` when no node metadata is
    available (i.e. every network but a metadata-enabled co-citation one).
    """
    if not node_meta:
        return []
    from pysyrev.core.networks import cluster_profiles

    profiles = cluster_profiles(result.node_ids, result.labels, result.W, node_meta)
    if not profiles:
        return []

    rows = []
    for comm in sorted(profiles):
        p = profiles[comm]
        span = (f"{p['year_range'][0]}–{p['year_range'][1]}"
                if p["year_range"] else "-")
        rows.append([
            f"C{comm}",
            f"{p['n_known']}/{p['n_refs']}",
            f"{p['median_year']} ({span})" if p["median_year"] else "-",
            ", ".join(name for name, _ in p["authors"][:6]) or "-",
            ", ".join(name for name, _ in p["venues"]) or "-",
        ])
    blocks = [{
        "type": "table",
        "title": "Community profiles (most co-cited authors and venues)",
        "headers": ["Community", "Resolved", "Median year (span)",
                    "Most co-cited authors", "Main venues"],
        "rows": rows,
        "col_widths": [1.3, 1.3, 2.4, 6.0, 5.0],
    }]

    # The handful of works each community is actually built on — the concrete
    # anchor behind the terms and the author counts.
    work_rows = []
    for comm in sorted(profiles):
        for w in profiles[comm]["works"][:3]:
            first = (w.get("authors") or ["—"])[0]
            work_rows.append([
                f"C{comm}",
                str(w.get("title") or "—")[:80],
                first,
                str(w.get("year") or "-"),
                f"{w['strength']:.3g}",
            ])
    if work_rows:
        blocks.append({
            "type": "table",
            "title": "Most co-cited works per community",
            "headers": ["Community", "Title", "First author", "Year", "Strength"],
            "rows": work_rows,
            "col_widths": [1.3, 7.4, 3.2, 1.0, 1.4],
        })
    return blocks


def _build_network_subsection(result, df, bertopic_results, topic_labels, *,
                              title, filename_prefix, node_kind, count_label,
                              k, color_mode, export_to, cluster_labels=None):
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

    from pysyrev.core.references import (
        _extract_ref_doi, _normalize_doi_value, _BARE_DOI_RE)

    meta = df.drop_duplicates("id").set_index("id") if df is not None else None

    def _meta(nid, col, default="-"):
        if meta is not None and nid in meta.index and col in meta.columns:
            v = meta.at[nid, col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return default

    def _ref_doi(tok):
        """Normalized DOI of a token (raw citation, doi.org URL, or bare DOI),
        or None when it carries no DOI (e.g. an opaque OpenAlex id)."""
        if tok is None or (isinstance(tok, float) and np.isnan(tok)):
            return None
        tok = str(tok).strip()
        if not tok:
            return None
        d = _extract_ref_doi(tok)
        if d:
            return _normalize_doi_value(d)
        if _BARE_DOI_RE.match(tok.lower()):
            return _normalize_doi_value(tok)
        return None

    # In-corpus works keyed by their normalized DOI → title, so a reference node
    # expressed as a DOI (canonical reference_keys) resolves to its corpus title.
    doi_to_title = {}
    if meta is not None and "title" in meta.columns and "doi" in meta.columns:
        for dval, tval in zip(meta["doi"], meta["title"]):
            nd = _ref_doi(dval)
            if nd and isinstance(tval, str) and tval.strip():
                doi_to_title[nd] = tval

    # Resolved metadata for extra-corpus reference nodes (co-citation only, and
    # only when the metadata step ran) — the corpus itself knows nothing of them.
    node_meta = getattr(result, "node_meta", None) or {}

    def _title_doi(nid):
        """(title, doi) for a node: from the corpus by id (documents and
        in-corpus cited references), else from the resolved reference metadata,
        else from the reference key's own DOI (title looked up when the cited
        work is in-corpus), else ('—', '—') for an unresolved opaque id."""
        if meta is not None and nid in meta.index:                # in corpus by id
            t = _meta(nid, "title", None)
            return (str(t)[:70] if t else "—", _ref_doi(_meta(nid, "doi", None)) or "—")
        entry = node_meta.get(nid) or {}
        d = _ref_doi(nid) or entry.get("doi")     # the key's own DOI, else the resolved one
        t = entry.get("title") or doi_to_title.get(d)
        return (str(t)[:70] if t else "—", d or "—")

    def _year(nid):
        """Publication year of a node — the corpus row's, else the resolved
        reference metadata's."""
        y = _meta(nid, "year", None)
        if y is None:
            y = (node_meta.get(nid) or {}).get("year")
        return y if y is not None else "-"

    strength = result.W.sum(axis=1)
    hover = []
    for nid, comm, tp, st in zip(result.node_ids, result.labels, topics, strength):
        title_txt, doi_txt = _title_doi(nid)
        tlabel = color_labels.get(int(tp), "—")
        authors = ", ".join((node_meta.get(nid) or {}).get("authors") or [])[:80]
        hover.append(
            f"<b>{title_txt}</b><br>"
            + (f"{authors}<br>" if authors else "")
            + f"DOI: {doi_txt}<br>Year: {_year(nid)}<br>"
            f"Community: {'C%d' % comm if comm >= 0 else 'tail'}<br>"
            f"Topic: {tlabel}<br>Strength: {st:.3g}"
        )

    color_by = topics if color_mode == "topic" else None
    fig = plot_network(
        result.coords, result.labels, result.W, k=k,
        color_by=color_by, color_labels=color_labels,
        hover_text=hover, title=None,
    )

    # ── Stats + top-10 nodes by strength ───────────────────────────────────
    n_backbone = len(backbone_edges(result.W, result.labels, k=k))
    n_full     = int((result.W > 0).sum() // 2)
    connected  = int((strength > 0).sum())
    kv_items = [
        {"key": count_label,               "value": result.n_nodes},
        {"key": f"Connected {node_kind}s", "value": f"{connected} ({connected / result.n_nodes:.0%})"},
        {"key": "Leiden communities",      "value": result.n_communities},
        {"key": "Modularity",              "value": f"{result.modularity:.3f}"},
    ]
    if getattr(result, "resolution", None) is not None:
        res_val = f"{result.resolution:.2f}"
        if getattr(result, "resolution_sweep", None):
            res_val += f" (selected by max modularity over {len(result.resolution_sweep)} values)"
        kv_items.append({"key": "Leiden resolution", "value": res_val})
    kv_items += [
        {"key": "Edges",                   "value": n_full},
        {"key": f"Backbone edges (k={k})", "value": n_backbone},
    ]
    sub_content = [{"type": "key_value", "items": kv_items}]

    order = np.argsort(strength)[::-1][:10]
    rows = []
    for i in order:
        nid = result.node_ids[i]
        comm = int(result.labels[i])
        title_txt, doi_txt = _title_doi(nid)
        rows.append([
            title_txt,
            doi_txt,
            f"{strength[i]:.3g}",
            f"C{comm}" if comm >= 0 else "-",
            color_labels.get(int(topics[i]), "-"),
        ])
    if rows:
        sub_content.append({
            "type": "table",
            "title": f"Top 10 {node_kind}s by strength",
            "headers": ["Title", "DOI", "Strength", "Community", "Topic"],
            "rows": rows,
            "col_widths": [6.0, 3.2, 1.4, 1.6, 3.8],
        })

    # Per-community TF-IDF sub-themes: each Leiden community's own thematic field,
    # independent of the BERTopic topics (the skill's network-first naming).
    terms = getattr(result, "terms", None)
    if terms:
        comm_size = {c: int((result.labels == c).sum()) for c in terms}
        use_llm = bool(cluster_labels)
        term_rows = []
        for c in sorted(terms):
            row = [f"C{c}", str(comm_size.get(c, 0))]
            if use_llm:
                row.append(str(cluster_labels.get(int(c), "—")))
            row.append(", ".join(terms[c][:10]))
            term_rows.append(row)
        size_header = "Docs" if node_kind == "document" else "Refs"
        if use_llm:
            headers    = ["Community", size_header, "LLM label", "Top terms"]
            col_widths = [1.4, 1.0, 4.6, 9.8]
        else:
            headers    = ["Community", size_header, "Top terms"]
            col_widths = [1.6, 1.2, 14.0]
        sub_content.append({
            "type": "table",
            "title": "Community sub-themes (distinguishing TF-IDF terms)",
            "headers": headers,
            "rows": term_rows,
            "col_widths": col_widths,
        })

    sub_content.extend(_community_profile_blocks(result, node_meta))

    if color_mode == "topic":
        caption = ("Layout comes from the coupling structure (strongly-coupled "
                   "papers sit together); node colour is the BERTopic topic, so "
                   "each community's topic mix is visible. Node size scales with "
                   "strength; grey nodes are unconnected.")
    else:
        caption = ("Layout and colour both come from the network structure "
                   "(Leiden communities on the coupling graph, so communities "
                   "read as spatially separated clusters). Node size scales with "
                   "strength.")
    sub_content.append({
        "type": "plotly", "figure": fig,
        "caption": caption, "filename_prefix": filename_prefix,
    })

    html_block = _export_network_html(fig, export_to, title)
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection", "title": title, "blocks": sub_content}


_UNIT_PLURALS = {"topic": "topics", "community": "communities"}


# Vocabulary of one connectivity panel. The matrix machinery is metric-agnostic —
# inter_cluster_matrix averages blocks of any weighted W — so only the prose
# differs between a coupling W (how much two groups cite the same works) and a
# co-citation W (how often two groups' references are cited together).
_CONNECTIVITY_KINDS = {
    "coupling": {
        "metric":   "bibliographic coupling",
        "basis":    "shared references",
        "colorbar": "mean coupling",
        "offdiag":  "how much two {units} share references",
        "darker":   "stronger shared-reference overlap",
        "profile":  ("A {unit} with high internal and low outward coupling (below "
                     "baseline) is a self-contained / weakly integrated theme."),
    },
    "cocitation": {
        "metric":   "co-citation",
        "basis":    "documents citing both references",
        "colorbar": "mean co-citation",
        "offdiag":  "how often the references of two {units} are cited together",
        "darker":   "more frequent co-citation",
        "profile":  ("A {unit} with high internal and low outward co-citation (below "
                     "baseline) is an intellectual base mobilised on its own, rarely "
                     "alongside the others."),
    },
}


def _connectivity_panel(W, groups, unit, scale, export_to,
                        kind="coupling", title=None):
    """One inter-group connectivity subsection for a precomputed grouping.

    *groups* is a list of ``(label, idx_array)``; *unit* is ``"topic"`` or
    ``"community"``; *kind* selects the metric's vocabulary from
    :data:`_CONNECTIVITY_KINDS` (``W`` is a coupling or a co-citation matrix).
    A heatmap of the mean weight between groups plus a per-group
    internal/outward table. Returns None when fewer than two non-empty groups
    exist.

    *title* overrides the subsection title; it also drives the exported HTML
    filename, so two panels of the same *unit* over different networks must not
    share one.
    """
    from pysyrev.core.networks import (
        inter_cluster_matrix, corpus_baseline, coupling_inout,
    )
    from pysyrev.core.figures import plot_connectivity_matrix

    groups = [(lab, idx) for lab, idx in groups if len(idx) > 0]
    if len(groups) < 2:
        return None

    words  = _CONNECTIVITY_KINDS[kind]
    title  = title or f"{unit.capitalize()} connectivity"
    units  = _UNIT_PLURALS.get(unit, unit + "s")

    M, labs = inter_cluster_matrix(W, groups, scale=scale)
    base    = corpus_baseline(W, scale=scale)
    fig     = plot_connectivity_matrix(M, labs, baseline=base, title=None,
                                       colorbar_title=words["colorbar"])

    rows = []
    for i, lab in enumerate(labs):
        p = coupling_inout(M, i)
        siloed = p["internal"] > p["outward"] and p["outward"] < base
        rows.append([lab, f"{p['internal']:.2f}", f"{p['outward']:.2f}",
                     "self-contained" if siloed else "integrated"])

    sub_content = [
        {"type": "paragraph",
         "text": (f"Mean {words['metric']} between {units} ({words['basis']}, "
                  f"scaled ×{int(scale)}). The diagonal is each {unit}'s internal "
                  f"cohesion; off-diagonal cells show "
                  f"{words['offdiag'].format(unit=unit, units=units)}. "
                  f"Corpus baseline: <b>{base:.2f}</b>.")},
        {"type": "plotly", "figure": fig,
         "caption": (f"Darker = {words['darker']}. "
                     + words["profile"].format(unit=unit, units=units)),
         "filename_prefix": _slug(title)},
        {"type": "table",
         "title": f"Internal vs outward {words['metric']} per {unit}",
         "headers": [unit.capitalize(), "Internal", "Outward", "Profile"],
         "rows": rows, "col_widths": [7.0, 3.0, 3.0, 4.0]},
    ]
    html_block = _export_network_html(fig, export_to, title)
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection", "title": title, "blocks": sub_content}


def _build_connectivity_subsections(coupling_result, cocitation_result,
                                    bertopic_results, topic_labels, section_cfg,
                                    export_to):
    """The connectivity panels — by BERTopic topic, then by Leiden community for
    each network that has one.

    Topic connectivity tests whether the global topics share references. Community
    connectivity asks the same of the communities themselves, once per network:
    for coupling, whether two groups of papers cite the same works; for
    co-citation, whether two intellectual bases are mobilised together by the same
    documents. Both readings matter and they are not interchangeable, so they are
    rendered as sibling panels rather than one.

    There is deliberately no topic panel for co-citation: its nodes are
    references, mostly extra-corpus, which carry no BERTopic topic.

    Returns a list of subsection blocks (0-3), skipping any grouping with fewer
    than two non-empty groups.
    """
    from pysyrev.core.networks import groups_from_labels

    scale = section_cfg.connectivity.scale
    panels = []

    # By BERTopic topic — coupling only (topics are a property of documents).
    if coupling_result is not None and coupling_result.n_nodes >= 2:
        topic_of = {}
        if bertopic_results is not None and "Topic" in bertopic_results.columns:
            id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
            if id_col:
                topic_of = dict(zip(bertopic_results[id_col], bertopic_results["Topic"]))
        if topic_of:
            topics = np.array([int(topic_of.get(nid, -1))
                               for nid in coupling_result.node_ids])
            tgroups = [(_topic_label(t, topic_labels), np.where(topics == t)[0])
                       for t in sorted(x for x in set(topics.tolist()) if x != -1)]
            panel = _connectivity_panel(coupling_result.W, tgroups, "topic", scale,
                                        export_to, kind="coupling")
            if panel is not None:
                panels.append(panel)

    # By Leiden community, once per network.
    for result, kind, label in ((coupling_result, "coupling", "coupling"),
                                (cocitation_result, "cocitation", "co-citation")):
        if result is None or result.n_nodes < 2:
            continue
        panel = _connectivity_panel(
            result.W, groups_from_labels(result.labels), "community", scale,
            export_to, kind=kind,
            title=f"Community connectivity ({label})")
        if panel is not None:
            panels.append(panel)

    return panels


def _build_community_topic_mapping_subsection(coupling_result, bertopic_results,
                                              topic_labels, export_to):
    """Mapping of Leiden coupling communities (rows) onto BERTopic topics (cols).

    A heatmap of how each coupling community's documents distribute across the
    global topics — the croisement between the citation-based communities and the
    text-based topics. Returns a subsection block, or None when a prerequisite is
    missing.
    """
    from pysyrev.core.networks import community_topic_crosstab
    from pysyrev.core.figures import plot_crosstab_heatmap

    if coupling_result is None or coupling_result.n_nodes < 2:
        return None
    if bertopic_results is None or "Topic" not in bertopic_results.columns:
        return None
    id_col = next((c for c in ["id", "ID"] if c in bertopic_results.columns), None)
    if id_col is None:
        return None

    topic_of = dict(zip(bertopic_results[id_col], bertopic_results["Topic"]))
    topics = np.array([int(topic_of.get(nid, -1)) for nid in coupling_result.node_ids])
    labels = np.asarray(coupling_result.labels)

    # Keep real communities (not the uncoupled tail) and real topics (not outliers).
    keep = (labels >= 0) & (topics >= 0)
    if keep.sum() < 1:
        return None
    ct = community_topic_crosstab(
        labels[keep], topics[keep],
        topic_labels={t: _topic_label(t, topic_labels)
                      for t in set(topics[keep].tolist())})
    if ct.empty or ct.shape[0] < 1 or ct.shape[1] < 1:
        return None

    row_labels = [f"C{c}" for c in ct.index]
    col_labels = [str(c) for c in ct.columns]
    fig = plot_crosstab_heatmap(
        ct.values, row_labels, col_labels, title=None,
        row_title="Coupling community", col_title="Topic")

    sub_content = [
        {"type": "paragraph",
         "text": ("How each bibliographic-coupling community (rows) distributes "
                  "across the global BERTopic topics (columns). A community "
                  "concentrated in a single topic is thematically focused; one "
                  "spread across several topics bridges them through shared "
                  "references.")},
        {"type": "plotly", "figure": fig,
         "caption": "Cell = number of documents; darker = more documents.",
         "filename_prefix": "community_topic_mapping"},
    ]
    html_block = _export_network_html(fig, export_to, "Community × topic mapping")
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection", "title": "Community × topic mapping",
            "blocks": sub_content}


def _build_networks_section(df, coupling_result, cocitation_result,
                            bertopic_results, topic_labels, section_cfg,
                            export_to, section_n, cluster_labels=None,
                            cocitation_labels=None):
    """Section — bibliographic coupling and co-citation networks.

    Coupling and co-citation are recomputed from the reviewed dataset's raw
    references (Salton / co-citation matrix → Leiden communities → backbone
    layout) and rendered with Plotly; coupling is coloured by BERTopic topic (the
    croisement). Connectivity panels follow: one by topic, then one per network by
    Leiden community (see :func:`_build_connectivity_subsections`). Citation is
    intentionally not rendered yet.
    """
    cfg = section_cfg
    sub_blocks = []

    coupling_sub = _build_network_subsection(
        coupling_result, df, bertopic_results, topic_labels,
        title="Bibliographic coupling", filename_prefix="bibliographic_coupling",
        node_kind="document", count_label="Documents",
        k=cfg.coupling.backbone_k, color_mode=cfg.coupling.color_by,
        export_to=export_to, cluster_labels=cluster_labels,
    )
    if coupling_sub is not None:
        sub_blocks.append(coupling_sub)

    mapping_sub = _build_community_topic_mapping_subsection(
        coupling_result, bertopic_results, topic_labels, export_to)
    if mapping_sub is not None:
        sub_blocks.append(mapping_sub)

    cocitation_sub = _build_network_subsection(
        cocitation_result, df, bertopic_results, topic_labels,
        title="Co-citation", filename_prefix="co_citation",
        node_kind="reference", count_label="References",
        k=cfg.cocitation.backbone_k, color_mode=cfg.cocitation.color_by,
        export_to=export_to, cluster_labels=cocitation_labels,
    )
    if cocitation_sub is not None:
        sub_blocks.append(cocitation_sub)

    sub_blocks.extend(_build_connectivity_subsections(
        coupling_result, cocitation_result, bertopic_results, topic_labels,
        cfg, export_to))

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
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim_fn
    from pysyrev.core.figures import plot_similarity_matrix
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

    selected_parts = []
    for topic_id in sorted(br_wo["Topic"].unique()):
        topic_df = br_wo[br_wo["Topic"] == topic_id].copy()
        filt     = topic_df[topic_df["year"] >= sel_cfg.min_year]
        if front_ids:                       # historical list = non-front papers only
            filt = filt[~filt[id_col].isin(front_ids)]

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
                      coupling_dataset: str = None,
                      cluster_labeler_config=None) -> dict:
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
    if not coupling_dataset:
        print("[report] networks skipped: no coupling dataset was provided "
              "(topic_model.doc_dataset is empty/unresolved) — the networks, "
              "connectivity and composite sections will be absent.")
    if coupling_dataset:
        try:
            _networks_df = pd.read_csv(coupling_dataset, low_memory=False)
        except Exception as e:
            print(f"[report] networks skipped: cannot read {coupling_dataset!r} ({e!r})")
            _networks_df = None
        if _networks_df is not None and not {"references", "id"}.issubset(_networks_df.columns):
            print(f"[report] networks skipped: dataset has no 'references'/'id' "
                  f"columns (found {list(_networks_df.columns)}).")
            _networks_df = None
        if _networks_df is not None:
            from pysyrev.core.networks import build_coupling, build_cocitation
            nc = sec.bib_network
            # Prefer the canonical DOI-keyed references when available: they
            # unify references across merged sources (see reference_keys). Fall
            # back to the raw references column otherwise.
            ref_col = ("reference_keys"
                       if "reference_keys" in _networks_df.columns
                       else "references")
            n_with_refs = int(_networks_df[ref_col].apply(
                lambda v: isinstance(v, str) and bool(v.strip())).sum())
            if n_with_refs < 2:
                print(f"[report] networks likely empty: only {n_with_refs} document(s) "
                      f"carry references in column {ref_col!r}.")
            try:
                _coupling_result = build_coupling(
                    _networks_df, ref_col=ref_col,
                    resolution_range=nc.coupling.resolution_range,
                    resolution_step=nc.coupling.resolution_step,
                    min_size=nc.coupling.min_size)
            except Exception as e:
                print(f"[report] coupling network skipped: {e!r}")
                _coupling_result = None
            try:
                _cocitation_result = build_cocitation(
                    _networks_df, ref_col=ref_col, min_ref_freq=nc.cocitation.min_ref_freq,
                    resolution_range=nc.cocitation.resolution_range,
                    resolution_step=nc.cocitation.resolution_step,
                    min_size=nc.cocitation.min_size,
                    ref_meta=_cocitation_metadata(_networks_df, ref_col, nc.cocitation))
            except Exception as e:
                print(f"[report] co-citation network skipped: {e!r}")
                _cocitation_result = None

    # Human-readable labels for the communities of both networks, via the same
    # LLM used for topics. Computed once and cached (keyed on a hash of the
    # community sub-theme terms), so the labeller never re-runs on an unchanged
    # partition. Co-citation only has terms when its metadata step ran.
    _cluster_labels = _label_coupling_communities(
        run_dir, _coupling_result, _networks_df, cluster_labeler_config)
    _cocitation_labels = _label_cocitation_communities(
        run_dir, _cocitation_result, cluster_labeler_config)

    # 2. Bibliographic networks (coupling + co-citation) — rendered whenever the
    # reviewed dataset yielded a coupling network (i.e. it carried references).
    if _coupling_result is not None:
        section = _build_networks_section(
            _networks_df, _coupling_result, _cocitation_result,
            bertopic_results, topic_labels, sec.bib_network, export_to, n,
            cluster_labels=_cluster_labels,
            cocitation_labels=_cocitation_labels)
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
