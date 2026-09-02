import numpy as np


def inter_cluster_matrix(W, groups, scale=1000.0):
    """Mean bibliographic-coupling between every pair of clusters.

    W      : symmetric (N,N) coupling matrix (e.g. Salton-normalised shared refs).
    groups : list of (label, idx_array) — each idx_array holds the row/col
             positions in W for that cluster. Order defines the matrix order.
    scale  : multiply every cell by this (coupling values are tiny; 1000 reads well).

    Returns (M, labels):
      M[i,j] = mean coupling of the block W[group_i, group_j].
        - diagonal (i==j): mean over the STRICT upper triangle of the within-
          cluster block, i.e. every distinct intra-cluster pair, excluding the
          zero self-diagonal. This is the cluster's internal cohesion.
        - off-diagonal (i!=j): plain mean of the full rectangular block.
      labels = [label for label,_ in groups].
    Symmetric by construction (M[i,j]==M[j,i]).
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
                M[i, j] = blk.mean()
    return M * scale, [lab for lab, _ in groups]


def corpus_baseline(W, scale=1000.0):
    """Mean coupling over all distinct pairs in the whole matrix — the
    reference line to judge whether an inter-cluster block is high or low."""
    N = W.shape[0]
    iu = np.triu_indices(N, 1)
    return float(W[iu].mean() * scale)


def coupling_inout(M, i):
    """From an inter-cluster matrix M, the inward/outward profile of group i:
      internal = M[i,i]  (within-cluster cohesion)
      outward  = mean of the off-diagonal cells on row i (coupling to the rest)
      per_target = dict {j: M[i,j]} for j != i
    A group with internal >> outward is self-contained / weakly integrated."""
    G = M.shape[0]
    others = [j for j in range(G) if j != i]
    return {
        "internal": float(M[i, i]),
        "outward": float(np.mean([M[i, j] for j in others])) if others else 0.0,
        "per_target": {j: float(M[i, j]) for j in others},
    }
