"""
Bibliographic-coupling network — the coupling-specific matrix + build.

Two documents are coupled when they cite the same references; the more they
share (relative to their reference-list sizes), the stronger the link. This
module holds only what is specific to coupling — the Salton-normalised
document×document matrix — and composes the shared toolkit in :mod:`.common`
(reference sets → Leiden communities → force layout) into a single
:class:`~pysyrev.core.networks.common.NetworkResult`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from pysyrev.core.bib import REFS
from pysyrev.core.networks.common import (
    NetworkResult,
    force_layout,
    leiden_best_resolution,
    leiden_communities,
    reference_sets,
    resolution_grid,
)


def salton_coupling(refsets: Sequence[set]) -> np.ndarray:
    """Salton-normalised bibliographic-coupling matrix.

    ``W[i,j] = shared / sqrt(|refs_i| * |refs_j|)`` — the cosine-like
    normalisation stops reference-rich reviews from dominating every edge.
    Returns a symmetric ``(N, N)`` matrix with a zero diagonal.
    """
    n = len(refsets)
    W = np.zeros((n, n))
    for i in range(n):
        a = refsets[i]
        if not a:
            continue
        for j in range(i + 1, n):
            b = refsets[j]
            if not b:
                continue
            shared = len(a & b)
            if shared:
                W[i, j] = W[j, i] = shared / np.sqrt(len(a) * len(b))
    return W


def build_coupling(df, ref_col: str = REFS, resolution: float = 0.5,
                   seed: int = 7, min_size: int = 5,
                   layout_niter: int = 500,
                   resolution_range: Optional[Sequence[float]] = None,
                   resolution_step: float = 0.1) -> NetworkResult:
    """Build the full coupling network from a bib DataFrame.

    Pipeline: reference sets → Salton matrix → Leiden communities → force
    layout. Returns a :class:`NetworkResult` bundling node IDs, ``W``, community
    labels, coordinates, and modularity — ready to inspect or ``.plot()``.

    Parameters
    ----------
    df : pd.DataFrame
        Bib DataFrame (needs ``id`` and *ref_col*).
    ref_col : str
        Raw references column (OpenAlex IDs or WoS strings).
    resolution, seed, min_size :
        Passed to :func:`leiden_communities`. *resolution* is used only when no
        *resolution_range* is given.
    layout_niter : int
        Fruchterman-Reingold iterations for :func:`force_layout`.
    resolution_range : (min, max), optional
        When given, sweep Leiden resolutions from *min* to *max* (step
        *resolution_step*) and keep the partition with the best community
        structure (see :func:`leiden_best_resolution`); the layout still runs once,
        on the winner. The chosen resolution and the sweep are recorded on the
        result. When None, the fixed *resolution* is used.
    """
    node_ids, refsets = reference_sets(df, ref_col=ref_col)
    W = salton_coupling(refsets)
    if resolution_range is not None:
        labels, modularity, resolution, sweep = leiden_best_resolution(
            W, resolution_grid(resolution_range, resolution_step),
            seed=seed, min_size=min_size)
    else:
        labels, modularity = leiden_communities(
            W, resolution=resolution, seed=seed, min_size=min_size)
        sweep = None
    coords = force_layout(W, seed=seed, niter=layout_niter)
    return NetworkResult(node_ids=node_ids, W=W, labels=labels,
                         coords=coords, modularity=modularity,
                         resolution=resolution, resolution_sweep=sweep)
