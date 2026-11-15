"""
Plotly rendering for the reworked bibliographic networks.

Kept apart from :mod:`.common` (pure analysis) so the rendering dependency
(plotly) stays isolated. Plotly is used throughout so the report can embed both
a static PNG (for the PDF) and an interactive HTML graph.

Every network type shares this drawing code — the layout and communities it
visualises are produced upstream in :mod:`.common`.

The matrix panels that accompany these graphs (connectivity, community × topic
crosstab) are not network drawings and live in :mod:`pysyrev.core.report.figures`,
with every other heatmap of the report.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from pysyrev.core.networks.common import backbone_edges

# Categorical palette shared with the rest of the report.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_TAIL_COLOR = "#D6D6D6"
_EDGE_COLOR = "#CFCFCF"
# Continuous edge scale: a mid grey, so the opacity ramp has room to show.
_EDGE_COLOR_STRONG = "#8A8A8A"


def _node_sizes(W: np.ndarray, size_min: float = 6.0, size_max: float = 22.0,
                exponent: float = 1.0, values: Optional[np.ndarray] = None,
                scale: str = "log") -> np.ndarray:
    """Marker diameters from *values* (default: weighted degree, i.e. coupling
    strength).

    ``scale="log"`` compresses the range (log1p) — right for a strength whose
    tail would otherwise dwarf everything. ``scale="sqrt"`` makes the marker
    *area* proportional to the value, so a node with twice the papers or
    citations reads twice as big — right for counts the reader compares.
    """
    v = np.asarray(W).sum(axis=1) if values is None else np.asarray(values, dtype=float)
    v = np.clip(np.nan_to_num(v), 0, None)
    lv = np.sqrt(v) if scale == "sqrt" else np.log1p(v)
    lmax = float(lv.max()) or 1.0
    return size_min + (size_max - size_min) * (lv / lmax) ** exponent


_EDGE_LEVELS = 8   # width steps of the continuous edge scale (one trace each)


def _edge_traces(coords, bb, values: Optional[np.ndarray] = None,
                 scale: str = "tertile", width_range: tuple = (0.5, 6.0)):
    """Backbone edges, thicker = stronger.

    ``scale="tertile"`` (default): three widths by tertile of the backbone
    weights — a rank, not a magnitude. ``scale="linear"`` / ``"sqrt"``: width
    proportional to the value (or its square root, for counts with a long
    tail) over *width_range*, where *values* (a matrix aligned to ``W``, e.g.
    raw co-signed paper counts) replaces the edge weight when given; widths
    are binned into :data:`_EDGE_LEVELS` steps, opacity rising with width so
    strong links stand out of the mesh.
    """
    import plotly.graph_objects as go

    if not bb:
        return []
    if scale not in ("tertile", "linear", "sqrt"):
        raise ValueError(f"edge scale must be tertile|linear|sqrt, got {scale!r}")
    if scale in ("linear", "sqrt"):
        v = np.array([float(values[a, b]) if values is not None else w for a, b, w in bb])
        v = np.clip(v, 0, None)
        if scale == "sqrt":
            v = np.sqrt(v)
        frac = v / (v.max() or 1.0)
        levels = np.rint(frac * (_EDGE_LEVELS - 1)).astype(int)
        lo, hi = width_range
        traces = []
        for lvl in range(_EDGE_LEVELS):
            pairs = [(a, b) for (a, b, _), l in zip(bb, levels) if l == lvl]
            if not pairs:
                continue
            t = lvl / (_EDGE_LEVELS - 1)
            ex, ey = [], []
            for a, b in pairs:
                ex += [coords[a, 0], coords[b, 0], None]
                ey += [coords[a, 1], coords[b, 1], None]
            traces.append(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(width=lo + (hi - lo) * t, color=_EDGE_COLOR_STRONG),
                opacity=0.25 + 0.45 * t, hoverinfo="none", showlegend=False,
            ))
        return traces
    weights = np.array([w for _, _, w in bb])
    # Tertile thresholds → 3 width levels; degenerate cases collapse gracefully.
    qs = np.quantile(weights, [1 / 3, 2 / 3]) if len(weights) >= 3 else [weights.max()] * 2
    buckets = {0.6: [], 1.3: [], 2.2: []}
    widths = sorted(buckets)
    for a, b, w in bb:
        lvl = 0 if w <= qs[0] else (1 if w <= qs[1] else 2)
        buckets[widths[lvl]].append((a, b))
    traces = []
    for width, pairs in buckets.items():
        if not pairs:
            continue
        ex, ey = [], []
        for a, b in pairs:
            ex += [coords[a, 0], coords[b, 0], None]
            ey += [coords[a, 1], coords[b, 1], None]
        traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=width, color=_EDGE_COLOR),
            opacity=0.45, hoverinfo="none", showlegend=False,
        ))
    return traces


def plot_network(coords: np.ndarray, labels: np.ndarray, W: np.ndarray,
                 k: int = 3, seed_mask: Optional[np.ndarray] = None,
                 title: Optional[str] = None,
                 color_by: Optional[np.ndarray] = None,
                 color_labels: Optional[dict] = None,
                 hover_text: Optional[List[str]] = None,
                 width: int = 900, height: int = 650,
                 size_by: Optional[np.ndarray] = None,
                 node_text: Optional[List[str]] = None,
                 size_scale: str = "log",
                 size_range: tuple = (6.0, 22.0),
                 edge_values: Optional[np.ndarray] = None,
                 edge_scale: str = "tertile",
                 edge_width_range: tuple = (0.5, 6.0),
                 edge_hover: Optional[Callable[[int, int, float], str]] = None,
                 node_opacity: Optional[float] = None,
                 community_labels: Optional[dict] = None):
    """Build a Plotly figure of a network: backbone edges under nodes.

    Two colouring modes:

    * **By community** (default, ``color_by=None``) — nodes coloured by their
      Leiden community; the ``-1`` tail is faint. ``community_labels`` maps a
      community to its legend name (default ``C<id>``).
    * **By external attribute** (``color_by`` = array row-aligned to *coords*,
      e.g. BERTopic topics) — nodes coloured by that attribute, so you read the
      *topic composition of each bibliographic community*. The spatial clusters
      still come from the coupling layout (strongly-coupled papers sit together).
      ``color_labels`` maps values to legend names.

    Node size scales with coupling strength (weighted degree, log). ``hover_text``
    (row-aligned to *coords*) sets per-node hover; ``seed_mask`` rings seed nodes.
    ``size_by`` (row-aligned values, e.g. paper counts) replaces the strength as
    the size driver; ``node_text`` (row-aligned, ``""`` for none) writes a label
    beside the nodes that carry one. ``size_scale`` (``"log"`` | ``"sqrt"``,
    area-proportional) and ``size_range`` (min, max diameter in px) set how
    those values map to marker size — see :func:`_node_sizes`. Edges likewise:
    ``edge_scale`` (``"tertile"`` | ``"sqrt"``), ``edge_values`` (a matrix
    aligned to *W* whose entries drive the width instead of *W*'s, e.g. raw
    co-signed paper counts) and ``edge_width_range`` — see :func:`_edge_traces`.
    Which edges are drawn (the backbone) is always decided on *W*.
    ``edge_hover(a, b, w)`` returns the hover text of the drawn edge between
    nodes *a* and *b* (weight *w*); it is shown on an invisible marker at the
    edge's midpoint — visible in the interactive HTML only.

    Nodes are always drawn above the edges. ``node_opacity`` sets their
    opacity; ``None`` keeps Plotly's default, which is 0.7 for markers sized
    per point (a "bubble" trace) — so the edges underneath show *through* the
    nodes. ``1.0`` hides them. Within each trace larger nodes are drawn first,
    so an opaque large node never covers a smaller one.

    Returns a ``plotly.graph_objects.Figure`` — ready for the report's ``plotly``
    block (static PNG in the PDF + interactive HTML export).
    """
    import plotly.graph_objects as go

    labels = np.asarray(labels)
    coords = np.asarray(coords)
    n = len(labels)
    sizes = _node_sizes(W, size_min=size_range[0], size_max=size_range[1],
                        values=size_by, scale=size_scale)
    hover = hover_text if hover_text is not None else [f"node {i}" for i in range(n)]

    bb = backbone_edges(W, labels, k=k)
    data = list(_edge_traces(coords, bb, values=edge_values, scale=edge_scale,
                             width_range=edge_width_range))
    if edge_hover is not None and bb:
        # Plotly lines carry no per-segment hover: an invisible marker at each
        # edge's midpoint does. Drawn before the nodes, so nodes keep priority.
        data.append(go.Scatter(
            x=[(coords[a, 0] + coords[b, 0]) / 2 for a, b, _ in bb],
            y=[(coords[a, 1] + coords[b, 1]) / 2 for a, b, _ in bb],
            mode="markers", marker=dict(size=7, color="rgba(0,0,0,0)"),
            hovertext=[edge_hover(a, b, w) for a, b, w in bb], hoverinfo="text",
            showlegend=False, name="links",
        ))

    def _marker_trace(idx, color, name):
        idx = sorted(idx, key=lambda i: -sizes[i])     # large first, small on top
        marker = dict(size=[sizes[i] for i in idx], color=color,
                      line=dict(width=0.5, color="white"))
        if node_opacity is not None:
            marker["opacity"] = node_opacity
        return go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=name,
            hovertext=[hover[i] for i in idx], hoverinfo="text", marker=marker,
        )

    if color_by is None:
        # ── Colour by Leiden community ──────────────────────────────────────
        tail = np.where(labels < 0)[0]
        if len(tail):
            data.append(_marker_trace(tail, _TAIL_COLOR, "uncoupled / tail"))
        for i, c in enumerate(sorted(x for x in set(labels.tolist()) if x >= 0)):
            m = np.where(labels == c)[0]
            name = (community_labels or {}).get(c, f"C{c}")
            data.append(_marker_trace(m, _PALETTE[i % len(_PALETTE)], name))
    else:
        # ── Colour by external attribute (e.g. BERTopic topic) ──────────────
        color_by = np.asarray(color_by)
        out = np.where(color_by == -1)[0]
        if len(out):
            data.append(_marker_trace(out, _TAIL_COLOR, "outlier / no topic"))
        values = [v for v in sorted(set(color_by.tolist())) if v != -1]
        for i, v in enumerate(values):
            m = np.where(color_by == v)[0]
            name = (color_labels or {}).get(v, f"Topic {v}")
            data.append(_marker_trace(m, _PALETTE[i % len(_PALETTE)], name))

    if seed_mask is not None:
        sm = np.where(np.asarray(seed_mask))[0]
        if len(sm):
            data.append(go.Scatter(
                x=coords[sm, 0], y=coords[sm, 1], mode="markers", name="seed",
                marker=dict(size=[sizes[i] + 8 for i in sm], color="rgba(0,0,0,0)",
                            line=dict(width=1.4, color="#111")),
                hoverinfo="none",
            ))

    if node_text is not None:
        lab = [i for i in range(n) if node_text[i]]
        if lab:
            data.append(go.Scatter(
                x=coords[lab, 0], y=coords[lab, 1], mode="text",
                text=[node_text[i] for i in lab], textposition="top center",
                textfont=dict(size=10, color="#333"),
                hoverinfo="none", showlegend=False,
            ))

    fig = go.Figure(data=data, layout=go.Layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        width=width, height=height,
        plot_bgcolor="white",
    ))
    return fig
