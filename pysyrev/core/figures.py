"""
Shared figure builders for the report.

:func:`matrix_heatmap` is the single layout every matrix panel of the report
goes through — inter-group connectivity, community × topic crosstab, topic
similarity. They differ only in what a cell means (mean coupling, a document
count, a cosine) and therefore in how it is formatted, hovered and labelled;
everything else — colour scale, annotated cells, labels on top, size derived
from the matrix shape — is decided here so the panels read as one family.

Add a new matrix panel by writing a thin wrapper below rather than a fresh
``go.Heatmap``: that is what keeps the report visually coherent.

Kept out of :mod:`pysyrev.core.networks` on purpose — a heatmap of topic
similarity is not a network figure, and sections that draw one should not have
to import from the networks package.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# House style, shared by every matrix panel.
_COLORSCALE     = "Blues"
_ANNOTATION_PT  = 10
_COLORBAR_WIDTH = 12

# Default size rule for a square matrix: a per-cell allowance plus room for the
# labels, floored so a 2×2 matrix is still readable.
_SQUARE_CELL    = (90, 80)
_SQUARE_PAD     = (220, 180)
_SQUARE_MINIMUM = (460, 420)


def matrix_heatmap(z: np.ndarray,
                   x_labels: Sequence[str],
                   y_labels: Sequence[str],
                   *,
                   text_format: Optional[Callable[[float], str]] = None,
                   hovertemplate: Optional[str] = None,
                   colorbar_title: str = "",
                   title: Optional[str] = None,
                   x_title: str = "",
                   y_title: str = "",
                   colorscale: str = _COLORSCALE,
                   zmin: Optional[float] = 0.0,
                   zmax: Optional[float] = None,
                   cell: Tuple[int, int] = _SQUARE_CELL,
                   pad: Tuple[int, int] = _SQUARE_PAD,
                   minimum: Tuple[int, int] = _SQUARE_MINIMUM):
    """Annotated Plotly heatmap in the report's house style.

    Parameters
    ----------
    z : array-like
        The matrix, ``z[i][j]`` at row *i* (``y_labels[i]``) and column *j*.
        Non-finite cells are drawn as gaps and left unannotated — blanking a
        diagonal that would otherwise flatten the colour scale is a legitimate
        use (see :func:`plot_similarity_matrix`).
    x_labels, y_labels : sequence of str
        Column and row labels. Columns are labelled along the **top** edge and
        rows run top-to-bottom, so a matrix reads like a table.
    text_format : callable, optional
        ``value -> str`` used to annotate each cell; ``None`` leaves cells bare.
        Non-finite values never reach it.
    hovertemplate : str, optional
        Plotly hover template. Give one whenever the raw number needs a unit or
        more precision than the annotation shows.
    colorbar_title, title, x_title, y_title : str, optional
        Labels. ``title`` is usually left to the report section, which puts the
        explanation in a paragraph above the figure instead.
    zmin, zmax : float, optional
        Colour-scale bounds. ``zmax=None`` lets the data set the upper bound, so
        contrast covers the range the matrix actually occupies rather than a
        theoretical maximum.
    cell, pad, minimum : (int, int)
        Size rule, as ``(width, height)`` pairs: the figure is
        ``max(minimum, cell * n + pad)`` on each axis, with *n* the number of
        columns (width) or rows (height). The defaults suit a square matrix;
        a rectangular one passes its own.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    z = np.asarray(z, dtype=float)
    n_rows, n_cols = z.shape

    text = None
    if text_format is not None:
        text = [["" if not np.isfinite(v) else text_format(v) for v in row]
                for row in z]

    heatmap_kwargs = dict(
        z=z, x=list(x_labels), y=list(y_labels),
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, thickness=_COLORBAR_WIDTH),
    )
    if text is not None:
        heatmap_kwargs.update(text=text, texttemplate="%{text}",
                              textfont=dict(size=_ANNOTATION_PT))
    if hovertemplate is not None:
        heatmap_kwargs['hovertemplate'] = hovertemplate

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))
    fig.update_layout(
        title=title,
        xaxis=dict(side="top", tickangle=-30, automargin=True, title=x_title),
        yaxis=dict(autorange="reversed", automargin=True, title=y_title),
        width  = max(minimum[0], cell[0] * n_cols + pad[0]),
        height = max(minimum[1], cell[1] * n_rows + pad[1]),
        margin=dict(l=20, r=20, t=70, b=20),
        plot_bgcolor="white",
    )
    return fig


def plot_connectivity_matrix(M: np.ndarray, labels: List[str],
                             baseline: Optional[float] = None,
                             title: Optional[str] = None,
                             colorbar_title: str = "mean coupling"):
    """Inter-group connectivity matrix.

    ``M[i,j]`` is the mean edge weight between groups *i* and *j* (diagonal =
    internal cohesion) — bibliographic coupling by default, hence the colorbar
    default, but the same figure serves any weighted network (co-citation, say):
    pass *colorbar_title* to name the metric being averaged. *baseline*, the
    corpus-wide mean, is appended to the title as the line against which a block
    reads high or low.
    """
    subtitle = f"  (corpus baseline {baseline:.2f})" if baseline is not None else ""
    return matrix_heatmap(
        M, labels, labels,
        text_format=lambda v: f"{v:.1f}",
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        colorbar_title=colorbar_title,
        title=(title + subtitle) if title else (subtitle.strip() or None),
    )


def plot_similarity_matrix(M: np.ndarray, labels: List[str],
                           title: Optional[str] = None,
                           colorbar_title: str = "cosine"):
    """Square similarity matrix — cosine between topic distributions, say.

    Cells holding ``NaN`` are drawn as gaps: callers blank the diagonal, whose
    self-similarity of 1 is uninformative and would flatten the colour scale.
    """
    return matrix_heatmap(
        M, labels, labels,
        text_format=lambda v: f"{v:.2f}",
        hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
        colorbar_title=colorbar_title,
        title=title,
    )


def plot_crosstab_heatmap(M: np.ndarray, row_labels: List[str],
                          col_labels: List[str], title: Optional[str] = None,
                          row_title: str = "", col_title: str = "",
                          colorbar_title: str = "docs"):
    """Rectangular contingency table — Leiden community × BERTopic topic, say.

    ``M[i,j]`` is a document count, annotated as an integer with empty cells
    left blank. Sizing is tuned for a wide, short matrix rather than a square one.
    """
    return matrix_heatmap(
        M, col_labels, row_labels,
        text_format=lambda v: f"{int(v)}" if v else "",
        hovertemplate="%{y} × %{x}: %{z:.0f} docs<extra></extra>",
        colorbar_title=colorbar_title,
        title=title, x_title=col_title, y_title=row_title,
        cell=(80, 70), pad=(240, 160), minimum=(520, 360),
    )
