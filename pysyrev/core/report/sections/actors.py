"""
The actors section: who drives the literature — countries, institutions, authors.

Built from the reviewed dataset's canonical ``actors`` column only (see
:mod:`pysyrev.core.actors`), so it reads the same whatever the source. Per
level: the most involved actors, then their collaboration network.

Like the other optional sections it degrades rather than propagates: a level
whose network cannot be built (too few actors, too sparse) keeps its ranking
and says why the graph is missing.
"""

from collections import Counter

import numpy as np
import pandas as pd

from pysyrev.core.actors import ACTORS, level_entities
from pysyrev.core.report.sections.networks import _export_network_html

_LEVEL_TITLES = {"country": "Countries", "institution": "Institutions", "author": "Authors"}
_LEVEL_NOUNS = {"country": "country", "institution": "institution", "author": "author"}
_PLURAL = {"country": "countries", "institution": "institutions", "author": "authors"}

_SORT_TEXT = {
    "papers":     "number of papers (full counting)",
    "fractional": "fractional count (each paper split among its {plural})",
    "citations":  "citations received by their papers",
}


_SIZE_TEXT = {
    "citations":  "the citations its papers received",
    "papers":     "its number of papers",
    "fractional": "its fractional paper count",
}


_EDGE_TEXT = {
    "count":  "the number of papers co-signed",
    "weight": "the {norm}-normalised link strength",
}


def _pct(x) -> str:
    return "-" if pd.isna(x) else f"{x:.0%}"


def _identity_share(df, level):
    """Share of a level's distinct actors keyed by an identifier rather than a
    name — the reliability of every count below. ``None`` for countries."""
    if level == "country":
        return None
    keys = {e["key"] for cell in df[ACTORS] for e in level_entities(cell, level)}
    if not keys:
        return None
    schemes = Counter(k.split(":", 1)[0] for k in keys)
    return 1 - schemes.get("name", 0) / len(keys), schemes


def _summary_blocks(df):
    from pysyrev.core.networks import collaboration_summary

    s = collaboration_summary(df)
    n = s["n_papers"] or 1
    items = [
        {"key": "Papers", "value": s["n_papers"]},
        {"key": "With authors", "value": f"{s['with_author']} ({s['with_author'] / n:.0%})"},
        {"key": "With institutions", "value": f"{s['with_institution']} ({s['with_institution'] / n:.0%})"},
        {"key": "With countries", "value": f"{s['with_country']} ({s['with_country'] / n:.0%})"},
        {"key": "Distinct authors / institutions / countries",
         "value": f"{s['n_author']} / {s['n_institution']} / {s['n_country']}"},
        {"key": "Mean authors per paper",
         "value": "-" if np.isnan(s["mean_authors"]) else f"{s['mean_authors']:.1f}"},
        {"key": "Single-author papers", "value": _pct(s["single_author_share"])},
        {"key": "Multi-institution papers", "value": _pct(s["multi_institution_share"])},
        {"key": "International papers (>= 2 countries)", "value": _pct(s["international_share"])},
    ]
    blocks = [
        {"type": "paragraph", "text": (
            "Who writes this literature, and with whom. Each paper's authors, "
            "institutions and countries come from the bibliographic sources "
            "(OpenAlex, Web of Science…) and are unified on shared identifiers "
            "(ORCID for people, ROR for institutions, ISO codes for countries). "
            "Shares below are computed among the papers that carry the level, "
            "not the whole corpus.")},
        {"type": "key_value", "items": items},
    ]
    notes, by_name = [], False
    for level in ("author", "institution"):
        share = _identity_share(df, level)
        if share is not None:
            ident, _ = share
            by_name |= ident < 1
            notes.append(f"{ident:.0%} of {_PLURAL[level]} are identified by "
                         f"{'ORCID or a source id' if level == 'author' else 'ROR or a source id'}")
    if notes:
        tail = (" The others are matched on their normalised name (surname + "
                "first initial for authors), so homonyms may be merged and "
                "spelling variants split — read their counts as indicative."
                if by_name else "")
        blocks.append({"type": "callout", "title": "Identity of actors",
                       "text": "; ".join(notes) + "." + tail})
    return blocks


def _ranking_table(df, level, cfg):
    from pysyrev.core.networks import rank_actors

    r = rank_actors(df, level, top=cfg.top_n, sort_by=cfg.sort_by)
    if r.empty:
        return None
    headers = ["#", "Name"] + (["Country"] if level == "institution" else []) + [
        "Papers", "Fractional", "Share", "Citations", "Cit./paper", "Active"]
    rows = []
    for i, row in enumerate(r.itertuples(index=False), 1):
        span = "-" if pd.isna(row.first_year) else f"{row.first_year}–{row.last_year}"
        line = [str(i), str(row.name)[:60]]
        if level == "institution":
            line.append(row.country or "-")
        line += [str(int(row.papers)), f"{row.fractional:.1f}", _pct(row.share),
                 str(int(row.citations)), f"{row.citations_per_paper:.1f}", span]
        rows.append(line)
    # cm, summing to the 17 cm text width; the numeric columns are sized to
    # their header (9 pt bold + 2 × 6 pt padding).
    widths = (([0.9, 3.1, 1.7] if level == "institution" else [0.9, 4.8]) +
              [1.5, 2.1, 1.4, 1.9, 2.0, 2.4])
    sort_text = _SORT_TEXT[cfg.sort_by].format(plural=_PLURAL[level])
    return {
        "type": "table",
        "title": f"Top {len(rows)} {_PLURAL[level]} by {sort_text}",
        "headers": headers, "rows": rows, "col_widths": widths,
    }


def _network_blocks(df, level, cfg, export_to):
    from pysyrev.core.networks import build_collaboration, plot_network, backbone_edges

    net = cfg.network
    try:
        result = build_collaboration(
            df, level, min_papers=net.min_papers, max_nodes=net.max_nodes,
            normalize=net.normalize, resolution_range=net.resolution_range,
            resolution_step=net.resolution_step, min_size=net.min_size)
    except Exception as e:
        return [{"type": "paragraph",
                 "text": f"No {_LEVEL_NOUNS[level]} collaboration network: {e}."}]
    if int((result.W > 0).sum()) == 0:
        return [{"type": "paragraph", "text": (
            f"No {_LEVEL_NOUNS[level]} collaboration network: none of the "
            f"{result.n_nodes} most productive {_PLURAL[level]} share a paper.")}]

    meta = result.node_meta
    papers = np.array([float(meta.get(k, {}).get("papers", 0) or 0) for k in result.node_ids])
    size = np.array([float(meta.get(k, {}).get(net.size_by, 0) or 0) for k in result.node_ids])
    names = [str(meta.get(k, {}).get("name") or k) for k in result.node_ids]
    # Name the largest nodes — the ones the eye goes to first.
    top = set(np.argsort(-size)[:net.n_labels].tolist())
    node_text = [names[i][:28] if i in top else "" for i in range(result.n_nodes)]
    hover = []
    for i, k in enumerate(result.node_ids):
        m = meta.get(k, {})
        comm = int(result.labels[i])
        hover.append(
            f"<b>{names[i]}</b><br>"
            + (f"Country: {m.get('country')}<br>" if m.get("country") else "")
            + f"Papers: {int(papers[i])} (fractional {float(m.get('fractional') or 0):.1f})<br>"
            f"Citations: {int(m.get('citations') or 0)}<br>"
            f"Community: {'C%d' % comm if comm >= 0 else 'tail'}")
    counts = result.counts

    def _edge_hover(a, b, w):
        n = f"Co-signed papers: {int(counts[a, b])}<br>" if counts is not None else ""
        return (f"<b>{names[a]}</b> ↔ <b>{names[b]}</b><br>{n}"
                f"{net.normalize.capitalize()} strength: {w:.3g}")

    fig = plot_network(result.coords, result.labels, result.W, k=net.backbone_k,
                       edge_hover=_edge_hover,
                       hover_text=hover, size_by=size, node_text=node_text,
                       size_scale="sqrt", size_range=tuple(net.size_range),
                       edge_values=(result.counts if net.edge_width_by == "count" else None),
                       edge_scale="sqrt", edge_width_range=tuple(net.edge_width_range))

    connected = int((result.W.sum(axis=1) > 0).sum())
    kv = [
        {"key": f"{_LEVEL_TITLES[level]} drawn",
         "value": f"{result.n_nodes} (>= {net.min_papers} papers, at most {net.max_nodes})"},
        {"key": "Connected", "value": f"{connected} ({connected / result.n_nodes:.0%})"},
        {"key": "Collaboration links", "value": int((result.W > 0).sum() // 2)},
        {"key": f"Backbone edges (k={net.backbone_k})",
         "value": len(backbone_edges(result.W, result.labels, k=net.backbone_k))},
        {"key": "Leiden communities", "value": result.n_communities},
        {"key": "Modularity", "value": f"{result.modularity:.3f}"},
    ]
    if result.resolution is not None:
        kv.append({"key": "Leiden resolution", "value": f"{result.resolution:.2f}"})
    blocks = [{"type": "key_value", "items": kv}]

    comm_rows = []
    for c in sorted({int(x) for x in result.labels if x >= 0}):
        idx = [i for i in range(result.n_nodes) if int(result.labels[i]) == c]
        idx.sort(key=lambda i: -papers[i])
        comm_rows.append([f"C{c}", str(len(idx)), str(int(papers[idx].sum())),
                          ", ".join(names[i] for i in idx[:6])])
    if comm_rows:
        blocks.append({
            "type": "table",
            "title": f"Collaboration communities (most productive {_PLURAL[level]} first)",
            "headers": ["Community", "Members", "Papers*", "Main members"],
            "rows": comm_rows, "col_widths": [2.3, 2.1, 1.8, 10.8],
        })
    title = f"{_LEVEL_TITLES[level]} collaboration"
    blocks.append({
        "type": "plotly", "figure": fig, "filename_prefix": f"actors_{level}_network",
        "caption": (f"Two {_PLURAL[level]} are linked when they sign papers together; "
                    f"layout, communities and the {net.backbone_k} links drawn per node "
                    f"follow the {net.normalize}-normalised link strength. Colour is the "
                    f"Leiden community, node area is proportional to "
                    f"{_SIZE_TEXT[net.size_by]} (the {net.n_labels} largest are named), "
                    f"and edge width grows with the square root of "
                    f"{_EDGE_TEXT[net.edge_width_by].format(norm=net.normalize)}. "
                    f"*Papers summed over members: a co-signed paper counts once per member."),
    })
    html = _export_network_html(fig, export_to, title)
    if html is not None:
        blocks.append(html)
    return blocks


def _build_actors_section(df, section_cfg, export_to, section_n):
    """Section — authors, institutions and countries of the literature.

    *df* is the reviewed dataset; returns ``None`` when it has no ``actors``
    column (a corpus built before the column existed) or no actor at all.
    """
    if df is None or ACTORS not in df.columns or df[ACTORS].isna().all():
        return None
    cfg = section_cfg
    blocks = _summary_blocks(df)
    for level in cfg.levels:
        sub = []
        table = _ranking_table(df, level, cfg)
        if table is None:
            continue
        sub.append(table)
        if cfg.network.enabled:
            sub.extend(_network_blocks(df, level, cfg, export_to))
        blocks.append({"type": "subsection", "title": _LEVEL_TITLES[level], "blocks": sub})
    return {"title": f"{section_n}. Actors: countries, institutions, authors",
            "blocks": blocks}
