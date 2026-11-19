"""
The bibliographic-network section: coupling, co-citation and connectivity.

Both network panels are drawn by one builder — they differ only in what the
similarity matrix counts and in the prose that names it — and both feed the
connectivity matrices that say which groups are bibliographically adjacent and
which are siloed.

Every enrichment here degrades rather than propagates: unresolved reference
metadata, a failed LLM community label or a missing export directory costs the
report a detail, never the section.
"""

import os
import re
from pathlib import Path

import numpy as np

from pysyrev.core.report.sections.common import _short_label, _topic_label


def _slug(label: str) -> str:
    """Filename-safe slug of a panel title.

    Panel titles carry punctuation that has no business in a filename — the ``×``
    of "Community × topic mapping", the parentheses of "Community connectivity
    (co-citation)". Collapse every run of non-alphanumerics into one underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _ratio(v: float) -> str:
    """An observed/expected ratio as table text.

    The values run from well under 1 to the hundreds, so a fixed number of
    decimals either loses the small end or pads the large one: two decimals
    below 10, none above.
    """
    return f"{v:.0f}" if v >= 10 else f"{v:.2f}"


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
        # The HTML report embeds the live figure itself.
        "only":  "pdf",
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
        labels_are_cacheable, load_cached_cluster_labels, save_cluster_labels)

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
        if labels_are_cacheable(labels):
            path = save_cluster_labels(run_dir, terms, labels)
            print(f"{kind.capitalize()}-community labels saved to: {path}")
        elif labels:
            print(f"[report] {kind}-community labels incomplete — used for "
                  f"this report but not cached.")
        return labels or None
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


def _ref_doi(tok):
    """Normalized DOI of a token (raw citation, doi.org URL, or bare DOI),
    or None when it carries no DOI (e.g. an opaque OpenAlex id)."""
    from pysyrev.core.references import (
        _extract_ref_doi, _normalize_doi_value, _BARE_DOI_RE)

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


def _community_name(comm: int, cluster_labels) -> str:
    """How a community is named on a figure: its id, and its label when the
    communities were labelled — the id is what the tables refer to."""
    label = (cluster_labels or {}).get(int(comm))
    return f"C{comm} · {_short_label(label, 40)}" if label else f"C{comm}"


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
                _ref_doi(w.get("key")) or _ref_doi(w.get("doi")) or "—",
            ])
    if work_rows:
        blocks.append({
            "type": "table",
            "title": "Most co-cited works per community",
            "headers": ["Community", "Title", "First author", "Year", "Strength", "DOI"],
            "rows": work_rows,
            "col_widths": [1.3, 6.3, 2.6, 1.0, 1.4, 3.4],
        })
    return blocks


def _build_network_subsection(result, df, bertopic_results, topic_labels, *,
                              title, filename_prefix, node_kind, count_label,
                              k, color_mode, export_to, cluster_labels=None,
                              edge_scale="tertile", edge_width_range=(0.4, 3.5)):
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
        if meta is not None and nid in meta.index and col in meta.columns:
            v = meta.at[nid, col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return default

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

    # Legend and hover name a community by its label, when it has one.
    community_names = {int(c): _community_name(int(c), cluster_labels)
                       for c in set(result.labels.tolist()) if c >= 0}

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
            f"Community: {community_names.get(int(comm), 'tail')}<br>"
            f"Topic: {tlabel}<br>Strength: {st:.3g}"
        )

    # Edge hover (HTML only): the two endpoints, the raw count behind the link
    # — shared references for coupling, co-citing documents for co-citation —
    # and the normalised strength the width and the layout use.
    counts = getattr(result, "counts", None)
    count_text = ("Shared references" if node_kind == "document"
                  else "Documents citing both")
    short = [_title_doi(nid)[0][:50] for nid in result.node_ids]

    def _edge_hover(a, b, w):
        n = (f"{count_text}: {int(counts[a, b])}<br>"
             if counts is not None and counts.size else "")
        return f"<b>{short[a]}</b><br>↔ <b>{short[b]}</b><br>{n}Strength: {w:.3g}"

    color_by = topics if color_mode == "topic" else None
    fig = plot_network(
        result.coords, result.labels, result.W, k=k,
        color_by=color_by, color_labels=color_labels,
        hover_text=hover, title=None,
        edge_scale=edge_scale, edge_width_range=tuple(edge_width_range),
        edge_hover=_edge_hover, community_labels=community_names,
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
# inter_cluster_enrichment scores blocks of any weighted W — so only the prose
# differs between a coupling W (how much two groups cite the same works) and a
# co-citation W (how often two groups' references are cited together).
_CONNECTIVITY_KINDS = {
    "coupling": {
        "metric":   "bibliographic coupling",
        "basis":    "shared references",
        "colorbar": "coupling obs/exp (log₁₀)",
        "offdiag":  "how much two {units} share references",
        "darker":   "stronger shared-reference overlap",
        "support":  "references shared by at least two documents of the {unit}",
        "profile":  ("A {unit} with internal coupling above 1 and outward coupling "
                     "below 1 is a self-contained / weakly integrated theme."),
    },
    "cocitation": {
        "metric":   "co-citation",
        "basis":    "documents citing both references",
        "colorbar": "co-citation obs/exp (log₁₀)",
        "offdiag":  "how often the references of two {units} are cited together",
        "darker":   "more frequent co-citation",
        "support":  "documents citing at least two references of the {unit}",
        "profile":  ("A {unit} with internal co-citation above 1 and outward "
                     "co-citation below 1 is an intellectual base mobilised on its "
                     "own, rarely alongside the others."),
    },
}

# Text for an internal ratio that is not reported for lack of evidence. The
# blank is deliberate: a ratio from two observations is not a small ratio, it is
# an unknown one, and a number there would rank as if it were known.
_NO_VALUE = "—"


def _connectivity_panel(W, groups, unit, export_to, kind="coupling",
                        title=None, observations=None, min_support=10):
    """One inter-group connectivity subsection for a precomputed grouping.

    *groups* is a list of ``(label, idx_array)``; *unit* is ``"topic"`` or
    ``"community"``; *kind* selects the metric's vocabulary from
    :data:`_CONNECTIVITY_KINDS` (``W`` is a coupling or a co-citation matrix).
    A heatmap of the observed/expected weight between groups plus a per-group
    internal/outward table. Returns None when fewer than two non-empty groups
    exist.

    Blocks are scored as an enrichment over a degree-preserving null rather than
    as a raw block mean, which is not comparable across group sizes (see
    :mod:`pysyrev.core.networks.connectivity`). A small saturated group still
    scores high, so *observations* — the node indices behind each edge-producing
    observation — is counted by
    :func:`~pysyrev.core.networks.connectivity.internal_support` into the
    evidence behind each internal ratio. Below *min_support* that ratio is left
    blank in the table and its diagonal cell is a gap in the heatmap: a ratio
    from two or three observations is not a small one, it is an unknown one, and
    printing it would let it rank against ratios that are actually known. Only
    the diagonal is withheld — a group's outward cells rest on their own, far
    larger, evidence. Omit *observations* and nothing is withheld.

    *title* overrides the subsection title; it also drives the exported HTML
    filename, so two panels of the same *unit* over different networks must not
    share one.
    """
    from pysyrev.core.networks import (
        inter_cluster_enrichment, coupling_inout, internal_support,
    )
    from pysyrev.core.report.figures import plot_connectivity_matrix

    groups = [(lab, idx) for lab, idx in groups if len(idx) > 0]
    if len(groups) < 2:
        return None

    words  = _CONNECTIVITY_KINDS[kind]
    title  = title or f"{unit.capitalize()} connectivity"
    units  = _UNIT_PLURALS.get(unit, unit + "s")

    M, labs = inter_cluster_enrichment(W, groups)
    support = internal_support(observations, groups) if observations else None
    thin    = ([s < min_support for s in support] if support is not None
               else [False] * len(labs))

    # Withhold the internal ratio of a thinly-evidenced group everywhere it
    # would be read as a value: a gap on the heatmap's diagonal, a blank in the
    # table. The profile verdict goes with it — it is derived from that ratio.
    profiles = []
    rows     = []
    for i, lab in enumerate(labs):
        p = coupling_inout(M, i)
        if thin[i]:
            internal, profile = _NO_VALUE, "insufficient"
        else:
            internal = _ratio(p["internal"])
            profile  = ("self-contained" if p["internal"] > 1.0 > p["outward"]
                        else "integrated")
        profiles.append(profile)
        row = [lab, str(len(groups[i][1])), internal, _ratio(p["outward"])]
        if support is not None:
            row.append(str(support[i]))
        row.append(profile)
        rows.append(row)

    M_shown = M.astype(float).copy()
    for i, is_thin in enumerate(thin):
        if is_thin:
            M_shown[i, i] = np.nan
    fig = plot_connectivity_matrix(M_shown, labs, baseline=1.0, title=None,
                                   colorbar_title=words["colorbar"], ratio=True)

    headers = [unit.capitalize(), "Size", "Internal", "Outward"]
    widths  = [4.4, 1.8, 2.6, 2.6]
    if support is not None:
        headers.append("Support")
        widths.append(2.2)
    headers.append("Profile")
    widths.append(3.4)

    explain = (
        f"Each block scores the {words['metric']} between {units} "
        f"({words['basis']}) as <b>observed / expected</b>: the block's mean "
        f"weight divided by what a degree-preserving null model predicts for "
        f"the same pairs. <b>1.00</b> is the expected level, above is enriched, "
        f"below depleted. The diagonal is each {unit}'s internal cohesion; "
        f"off-diagonal cells show "
        f"{words['offdiag'].format(unit=unit, units=units)}. "
        f"The raw mean is not shown: it falls with {unit} size, since most "
        f"pairs of a large {unit} carry no weight and those zeros dominate the "
        f"average, which would rank small {units} far above large ones for "
        f"arithmetic reasons alone."
    )
    if support is not None:
        explain += (
            f" <b>Support</b> counts the {words['support'].format(unit=unit)} — "
            f"the evidence behind the internal ratio. Below {min_support} that "
            f"ratio is left blank rather than printed: computed from a handful "
            f"of observations it is not a small number but an unknown one, and "
            f"showing it would let it rank against ratios that are known. The "
            f"outward figure stands, resting on its own evidence."
        )

    sub_content = [
        {"type": "paragraph", "text": explain},
        {"type": "plotly", "figure": fig,
         "caption": (f"Darker = {words['darker']} than expected; colour runs on "
                     f"log₁₀ of the ratio, and a blank diagonal cell is a "
                     f"{unit} with too little evidence to score. "
                     + words["profile"].format(unit=unit, units=units)),
         "filename_prefix": _slug(title)},
        {"type": "table",
         "title": f"Internal vs outward {words['metric']} per {unit} (obs/exp)",
         "headers": headers, "rows": rows, "col_widths": widths},
    ]
    html_block = _export_network_html(fig, export_to, title)
    if html_block is not None:
        sub_content.append(html_block)

    return {"type": "subsection", "title": title, "blocks": sub_content}


def _connectivity_observations(df, coupling_result, cocitation_result) -> dict:
    """Node indices behind every edge-producing observation, per network.

    :func:`~pysyrev.core.networks.connectivity.internal_support` counts the
    observations that touch at least two nodes of a group. What an observation
    *is* flips with the network, but it is the same reference sets read two ways:

    * coupling — nodes are documents, so an observation is one **reference** and
      the documents citing it (an inverted index of the reference sets);
    * co-citation — nodes are references, so an observation is one **document**
      and the reference nodes it cites.

    Returns ``{"coupling": [...], "cocitation": [...]}``, omitting a network
    whose result is missing. Anything unexpected in the dataset yields ``{}``:
    support is a column of the table, never a reason to lose the panel.
    """
    if df is None:
        return {}
    try:
        from pysyrev.core.networks import reference_sets

        ref_col = ("reference_keys" if "reference_keys" in df.columns
                   else "references")
        if ref_col not in df.columns or "id" not in df.columns:
            return {}
        doc_ids, refsets = reference_sets(df, ref_col=ref_col)

        out = {}
        if coupling_result is not None:
            # One observation per reference: the coupling nodes (documents)
            # citing it. Only references cited twice or more create an edge.
            pos = {nid: i for i, nid in enumerate(coupling_result.node_ids)}
            by_ref = {}
            for doc_id, refs in zip(doc_ids, refsets):
                i = pos.get(doc_id)
                if i is None:
                    continue
                for r in refs:
                    by_ref.setdefault(r, []).append(i)
            out["coupling"] = [v for v in by_ref.values() if len(v) > 1]

        if cocitation_result is not None:
            # One observation per document: the co-citation nodes (references)
            # it cites. The long tail dropped by min_ref_freq is not a node.
            pos = {nid: i for i, nid in enumerate(cocitation_result.node_ids)}
            docs = []
            for refs in refsets:
                hit = [pos[r] for r in refs if r in pos]
                if len(hit) > 1:
                    docs.append(hit)
            out["cocitation"] = docs

        return out
    except Exception as e:                # support is a nicety, not the panel
        print(f"[report] connectivity support skipped: {e!r}")
        return {}


def _build_connectivity_subsections(df, coupling_result, cocitation_result,
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

    Each panel also reports how much evidence backs a group's internal value, so
    *df* is read once into per-network observations (see
    :func:`_connectivity_observations`).

    Returns a list of subsection blocks (0-3), skipping any grouping with fewer
    than two non-empty groups.
    """
    from pysyrev.core.networks import groups_from_labels

    panels = []
    obs = _connectivity_observations(df, coupling_result, cocitation_result)
    min_support = getattr(section_cfg.connectivity, "min_support", 10)

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
            panel = _connectivity_panel(coupling_result.W, tgroups, "topic",
                                        export_to, kind="coupling",
                                        observations=obs.get("coupling"),
                                        min_support=min_support)
            if panel is not None:
                panels.append(panel)

    # By Leiden community, once per network.
    for result, kind, label in ((coupling_result, "coupling", "coupling"),
                                (cocitation_result, "cocitation", "co-citation")):
        if result is None or result.n_nodes < 2:
            continue
        panel = _connectivity_panel(
            result.W, groups_from_labels(result.labels), "community",
            export_to, kind=kind, observations=obs.get(kind),
            min_support=min_support,
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
    from pysyrev.core.report.figures import plot_crosstab_heatmap

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
        edge_scale=cfg.coupling.edge_scale,
        edge_width_range=cfg.coupling.edge_width_range,
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
        edge_scale=cfg.cocitation.edge_scale,
        edge_width_range=cfg.cocitation.edge_width_range,
    )
    if cocitation_sub is not None:
        sub_blocks.append(cocitation_sub)

    sub_blocks.extend(_build_connectivity_subsections(
        df, coupling_result, cocitation_result, bertopic_results, topic_labels,
        cfg, export_to))

    if not sub_blocks:
        return None
    from pysyrev.core.report.sections.guides import networks_guide
    return {"title": f"{section_n}. Bibliographic networks",
            "blocks": [networks_guide(cfg)] + sub_blocks}
