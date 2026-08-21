"""
Co-citation network — the co-citation-specific matrix + build.

Two references are co-cited when they appear together in the reference list of a
corpus document; the more documents cite both, the stronger the link. Mirrors
:mod:`.coupling` but over references instead of documents: a Salton-normalised
reference×reference matrix, then the shared :mod:`.common` toolkit (Leiden
communities → force layout).

Nodes are reference identifiers (OpenAlex ``referenced_works`` IDs, or WoS
reference strings). Only references cited by at least ``min_ref_freq`` documents
are kept — a co-citation matrix over every once-cited reference is mostly noise
and needlessly large.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

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


def cocitation_matrix(refsets: Sequence[set],
                      min_ref_freq: int = 2) -> Tuple[List, np.ndarray]:
    """Salton-normalised co-citation matrix over references.

    ``W[a,b] = cocited(a,b) / sqrt(freq(a) * freq(b))`` where ``cocited`` is the
    number of documents citing both references and ``freq`` is how many
    documents cite a reference. Only references cited by at least
    *min_ref_freq* documents are kept as nodes.

    Returns ``(ref_ids, W)`` — ``ref_ids[i]`` labels row ``i`` of ``W``.
    """
    freq: Counter = Counter()
    for refs in refsets:
        freq.update(refs)

    kept = [r for r, c in freq.items() if c >= min_ref_freq]
    index = {r: i for i, r in enumerate(kept)}
    kept_set = set(kept)
    n = len(kept)

    W = np.zeros((n, n))
    for refs in refsets:
        present = [r for r in refs if r in kept_set]
        for a, b in combinations(present, 2):
            ia, ib = index[a], index[b]
            W[ia, ib] += 1
            W[ib, ia] += 1

    if n:
        f = np.array([freq[r] for r in kept], dtype=float)
        denom = np.sqrt(np.outer(f, f))
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(denom > 0, W / denom, 0.0)
        np.fill_diagonal(W, 0.0)
    return kept, W


def build_cocitation(df, ref_col: str = REFS, min_ref_freq: int = 2,
                     resolution: float = 0.7, seed: int = 7, min_size: int = 5,
                     layout_niter: int = 500,
                     resolution_range: Optional[Sequence[float]] = None,
                     resolution_step: float = 0.1) -> NetworkResult:
    """Build the full co-citation network from a bib DataFrame.

    Pipeline: reference sets → co-citation matrix → Leiden communities → force
    layout. Returns a :class:`NetworkResult` whose ``node_ids`` are reference
    identifiers (not corpus documents). Returns an empty result when fewer than
    two references clear *min_ref_freq*.

    Parameters
    ----------
    df : pd.DataFrame
        Bib DataFrame (needs ``id`` and *ref_col*).
    ref_col : str
        Raw references column (OpenAlex IDs or WoS strings).
    min_ref_freq : int
        Minimum number of documents citing a reference for it to be a node.
    resolution, seed, min_size :
        Passed to :func:`leiden_communities`. *resolution* is used only when no
        *resolution_range* is given.
    layout_niter : int
        Fruchterman-Reingold iterations for :func:`force_layout`.
    resolution_range : (min, max), optional
        When given, sweep resolutions from *min* to *max* (step *resolution_step*)
        and keep the best-structured partition (see :func:`leiden_best_resolution`);
        the layout runs once, on the winner.
    """
    _, refsets = reference_sets(df, ref_col=ref_col)
    ref_ids, W = cocitation_matrix(refsets, min_ref_freq=min_ref_freq)

    if len(ref_ids) < 2:
        return NetworkResult(node_ids=ref_ids, W=W,
                             labels=np.full(len(ref_ids), -1, dtype=int),
                             coords=np.zeros((len(ref_ids), 2)), modularity=0.0)

    if resolution_range is not None:
        labels, modularity, resolution, sweep = leiden_best_resolution(
            W, resolution_grid(resolution_range, resolution_step),
            seed=seed, min_size=min_size)
    else:
        labels, modularity = leiden_communities(
            W, resolution=resolution, seed=seed, min_size=min_size)
        sweep = None
    coords = force_layout(W, seed=seed, niter=layout_niter)
    return NetworkResult(node_ids=ref_ids, W=W, labels=labels,
                         coords=coords, modularity=modularity,
                         resolution=resolution, resolution_sweep=sweep)
