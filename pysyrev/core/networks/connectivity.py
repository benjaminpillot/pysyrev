"""
Inter-cluster connectivity — how groups of the coupling network connect.

Given the coupling matrix ``W`` and a grouping of its nodes (by BERTopic topic,
or by Leiden community), this measures the mean bibliographic-coupling weight
between every pair of groups: the diagonal is a group's internal cohesion (do
its papers share references?), the off-diagonal is how much two groups share
references (are they intellectually connected?), and the corpus baseline is the
line to judge any block as high or low.

Grouping by **topic** is the informative use: topics are not defined by
coupling, so their coupling connectivity is a genuine measurement (which topics
are bibliographically adjacent vs siloed). Grouping by Leiden **community** is
near-tautological — communities are defined to have high internal / low external
coupling — but is offered for completeness.

Ported from the ``cluster-connectivity-matrix`` skill.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def groups_from_labels(labels: Sequence[int],
                       prefix: str = "C") -> List[Tuple[str, np.ndarray]]:
    """Build ``(label, idx_array)`` groups from a per-node label array, one per
    label ``>= 0`` (the ``-1`` tail is dropped)."""
    labels = np.asarray(labels)
    return [(f"{prefix}{c}", np.where(labels == c)[0])
            for c in sorted(set(labels.tolist())) if c >= 0]


def inter_cluster_matrix(W: np.ndarray, groups: Sequence[Tuple[str, np.ndarray]],
                         scale: float = 1000.0) -> Tuple[np.ndarray, List[str]]:
    """Mean bibliographic-coupling between every pair of groups.

    ``groups`` is a list of ``(label, idx_array)`` where ``idx_array`` holds the
    row/col positions in ``W`` for that group; its order defines the matrix order.

    Returns ``(M, labels)``:
      * ``M[i,j]`` (``i != j``) — plain mean of the block ``W[group_i, group_j]``.
      * ``M[i,i]`` — mean over the strict upper triangle of the within-group
        block (every distinct intra-group pair, excluding the zero self-diagonal):
        the group's internal cohesion.
    ``scale`` only rescales for readability (coupling values are tiny); ratios and
    the baseline comparison are unaffected. Symmetric by construction.
    """
    G = len(groups)
    M = np.zeros((G, G))
    for i, (_, ai) in enumerate(groups):
        ai = np.asarray(ai)
        for j, (_, bj) in enumerate(groups):
            bj = np.asarray(bj)
            blk = W[np.ix_(ai, bj)]
            if i == j:
                iu = np.triu_indices(len(ai), 1)
                M[i, j] = blk[iu].mean() if len(iu[0]) else 0.0
            else:
                M[i, j] = blk.mean() if blk.size else 0.0
    return M * scale, [lab for lab, _ in groups]


def corpus_baseline(W: np.ndarray, scale: float = 1000.0) -> float:
    """Mean coupling over all distinct pairs in the whole matrix — the reference
    line to judge whether an inter-cluster block is high or low."""
    n = W.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, 1)
    return float(W[iu].mean() * scale)


def coupling_inout(M: np.ndarray, i: int) -> dict:
    """Inward/outward profile of group ``i`` from an inter-cluster matrix ``M``:

      * ``internal`` = ``M[i,i]`` (within-group cohesion),
      * ``outward``  = mean of the off-diagonal cells on row ``i`` (coupling to
        the rest),
      * ``per_target`` = ``{j: M[i,j]}`` for ``j != i``.

    A group with ``internal >> outward`` is self-contained / weakly integrated.
    """
    G = M.shape[0]
    others = [j for j in range(G) if j != i]
    return {
        "internal": float(M[i, i]),
        "outward": float(np.mean([M[i, j] for j in others])) if others else 0.0,
        "per_target": {j: float(M[i, j]) for j in others},
    }
