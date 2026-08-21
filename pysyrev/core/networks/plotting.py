"""
Plotly rendering for the reworked bibliographic networks.

Kept apart from :mod:`.common` (pure analysis) so the rendering dependency
(plotly) stays isolated. Plotly is used throughout so the report can embed both
a static PNG (for the PDF) and an interactive HTML graph.

Every network type shares this drawing code — the layout and communities it
visualises are produced upstream in :mod:`.common`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pysyrev.core.networks.common import backbone_edges

# Categorical palette shared with the rest of the report.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_TAIL_COLOR = "#D6D6D6"
_EDGE_COLOR = "#CFCFCF"


def _node_sizes(W: np.ndarray, size_min: float = 6.0, size_max: float = 22.0,
                exponent: float = 1.0) -> np.ndarray:
    """Marker sizes from weighted degree (coupling strength), log-scaled."""
    strength = np.asarray(W).sum(axis=1)
    lv = np.log1p(strength)
    lmax = float(lv.max()) or 1.0
    return size_min + (size_max - size_min) * (lv / lmax) ** exponent


def _edge_traces(coords, bb):
    """Backbone edges as up to three width buckets (thicker = stronger)."""
    import plotly.graph_objects as go

    if not bb:
        return []
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
                 width: int = 900, height: int = 650):
    """Build a Plotly figure of a network: backbone edges under nodes.

    Two colouring modes:

    * **By community** (default, ``color_by=None``) — nodes coloured by their
      Leiden community; the ``-1`` tail is faint.
    * **By external attribute** (``color_by`` = array row-aligned to *coords*,
      e.g. BERTopic topics) — nodes coloured by that attribute, so you read the
      *topic composition of each bibliographic community*. The spatial clusters
      still come from the coupling layout (strongly-coupled papers sit together).
      ``color_labels`` maps values to legend names.

    Node size scales with coupling strength (weighted degree, log). ``hover_text``
    (row-aligned to *coords*) sets per-node hover; ``seed_mask`` rings seed nodes.

    Returns a ``plotly.graph_objects.Figure`` — ready for the report's ``plotly``
    block (static PNG in the PDF + interactive HTML export).
    """
    import plotly.graph_objects as go

    labels = np.asarray(labels)
    coords = np.asarray(coords)
    n = len(labels)
    sizes = _node_sizes(W)
    hover = hover_text if hover_text is not None else [f"node {i}" for i in range(n)]

    data = list(_edge_traces(coords, backbone_edges(W, labels, k=k)))

    def _marker_trace(idx, color, name):
        return go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=name,
            hovertext=[hover[i] for i in idx], hoverinfo="text",
            marker=dict(size=[sizes[i] for i in idx], color=color,
                        line=dict(width=0.5, color="white")),
        )

    if color_by is None:
        # ── Colour by Leiden community ──────────────────────────────────────
        tail = np.where(labels < 0)[0]
        if len(tail):
            data.append(_marker_trace(tail, _TAIL_COLOR, "uncoupled / tail"))
        for i, c in enumerate(sorted(x for x in set(labels.tolist()) if x >= 0)):
            m = np.where(labels == c)[0]
            data.append(_marker_trace(m, _PALETTE[i % len(_PALETTE)], f"C{c}"))
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


def plot_connectivity_matrix(M: np.ndarray, labels: List[str],
                             baseline: Optional[float] = None,
                             title: Optional[str] = None):
    """Annotated Plotly heatmap of an inter-group connectivity matrix.

    ``M[i,j]`` is the mean bibliographic coupling between groups *i* and *j*
    (diagonal = internal cohesion). Each cell is annotated with its value.
    Returns a ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    M = np.asarray(M, dtype=float)
    text = [[f"{v:.1f}" for v in row] for row in M]
    n = len(labels)

    fig = go.Figure(go.Heatmap(
        z=M, x=labels, y=labels,
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale="Blues", zmin=0.0,
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="mean coupling", thickness=12),
    ))
    subtitle = f"  (corpus baseline {baseline:.2f})" if baseline is not None else ""
    fig.update_layout(
        title=(title + subtitle) if title else (subtitle.strip() or None),
        xaxis=dict(side="top", tickangle=-30, automargin=True),
        yaxis=dict(autorange="reversed", automargin=True),
        width=max(460, 90 * n + 220), height=max(420, 80 * n + 180),
        margin=dict(l=20, r=20, t=70, b=20),
        plot_bgcolor="white",
    )
    return fig
