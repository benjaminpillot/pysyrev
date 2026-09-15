"""
Inter-cluster connectivity — how groups of the coupling network connect.

Given the coupling matrix ``W`` and a grouping of its nodes (by BERTopic topic,
or by Leiden community), this measures how much weight sits between every pair
of groups: the diagonal is a group's internal cohesion (do its papers share
references?) and the off-diagonal is how much two groups share references (are
they intellectually connected?).

Two readings of the same blocks are available:

* :func:`inter_cluster_matrix` — the **mean** weight over the block, judged
  against :func:`corpus_baseline`. Intuitive, but not comparable across group
  sizes: the mean of a sparse block falls as the block grows, since most pairs
  of a large group carry no weight and those zeros dominate the average.
* :func:`inter_cluster_enrichment` — the same block mean divided by what a
  degree-preserving null expects over the same pairs
  (:func:`configuration_null`). The pair count cancels, leaving one scale
  whatever the group size: 1.0 is the expected level, above enriched, below
  depleted. This corrects the arithmetic penalty on large groups, not a small
  group's genuine saturation — a few nodes always seen together really are
  enriched, they are merely thinly evidenced.

:func:`internal_support` measures that last point, counting the observations
that actually produce a group's internal weight.

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


def configuration_null(W: np.ndarray) -> np.ndarray:
    """Expected weight between every pair of nodes under a degree-preserving null.

    ``E[a,b] = s_a * s_b / 2m``, with ``s_a`` the strength of node ``a`` (its row
    sum in ``W``) and ``2m = sum(s)``: the weight one would expect between ``a``
    and ``b`` if the network's whole weight were redistributed at random while
    every node kept its own total strength. An all-zero ``W`` yields all zeros.
    """
    W = np.asarray(W, dtype=float)
    s = W.sum(axis=1)
    total = float(s.sum())
    if total <= 0:
        return np.zeros_like(W)
    return np.outer(s, s) / total


def inter_cluster_enrichment(
        W: np.ndarray,
        groups: Sequence[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    """Observed / expected weight between every pair of groups.

    ``R[i,j]`` divides the block mean of :func:`inter_cluster_matrix` by the
    block mean of :func:`configuration_null` over the *same* pairs, so the pair
    count cancels and what is left is judged against the strength the block's
    own nodes carry. That is what makes ``R`` comparable across group sizes
    where the raw mean is not. The baseline is ``1.0``: above is more weight
    than the node strengths predict, below is less.

    Returns ``(R, labels)``. Blocks whose expectation is zero yield ``0.0``.

    The null's block means are derived in closed form from the node strengths;
    averaging :func:`configuration_null` itself would materialise a second dense
    ``n x n`` matrix for a result that is a handful of numbers.
    """
    W = np.asarray(W, dtype=float)
    obs, labs = inter_cluster_matrix(W, groups, scale=1.0)

    s = W.sum(axis=1)
    total = float(s.sum())
    G = len(groups)
    exp = np.zeros((G, G))
    if total > 0:
        # Mean of E[a,b] = s_a*s_b/total over the same pairs inter_cluster_matrix
        # averages: the strict upper triangle within a group, the full block
        # between two (disjoint) groups.
        sums = [float(s[np.asarray(idx)].sum()) for _, idx in groups]
        sqs  = [float((s[np.asarray(idx)] ** 2).sum()) for _, idx in groups]
        for i, (_, ai) in enumerate(groups):
            n_i = len(ai)
            if n_i > 1:
                exp[i, i] = (sums[i] ** 2 - sqs[i]) / (total * n_i * (n_i - 1))
            for j in range(i + 1, G):
                n_j = len(groups[j][1])
                if n_i and n_j:
                    exp[i, j] = exp[j, i] = sums[i] * sums[j] / (total * n_i * n_j)

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(exp > 0, obs / exp, 0.0)
    return R, labs


def internal_support(observations: Sequence[Sequence[int]],
                     groups: Sequence[Tuple[str, np.ndarray]]) -> List[int]:
    """How many observations actually back each group's internal weight.

    An *observation* is whatever creates an edge, given as the node indices it
    touches: for co-citation, one corpus document and the references it cites;
    for coupling, one reference and the documents citing it. It supports a group
    when it touches at least **two** of its nodes — that is one internal pair.

    Returns one count per group, in ``groups`` order. A high internal weight on
    a support of two says little, whatever the metric.
    """
    out: List[int] = []
    for _, idx in groups:
        members = {int(i) for i in np.asarray(idx).ravel().tolist()}
        n = 0
        for obs in observations:
            hit = 0
            for node in obs:
                if node in members:
                    hit += 1
                    if hit >= 2:
                        n += 1
                        break
        out.append(n)
    return out
