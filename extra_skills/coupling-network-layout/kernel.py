import numpy as np


def coupling_layout(W, seed=7, niter=500, mode="fr"):
    """2-D coordinates for a bibliographic-coupling network, laid out so that
    strongly-coupled papers sit together — which is what makes Leiden
    communities read as visually separated blobs.

    W     : symmetric (N,N) coupling matrix. Edges are the non-zero entries;
            edge WEIGHT is the actual coupling value, so the force-directed
            layout pulls high-overlap papers together.
    seed  : RNG seed (both python `random` and numpy) for a reproducible layout.
    niter : Fruchterman-Reingold iterations (higher = tighter, slower).
    mode  : "fr" (Fruchterman-Reingold, weighted) or "drl" (scales better past
            a few thousand nodes).

    Returns coords : (N,2) float array, row-aligned to W.
    Nodes with no edge float free; drop them from the plot or let FR park them
    at the rim.
    """
    import random as pyr
    import igraph as ig
    A = (np.asarray(W) > 0).astype(float)
    g = ig.Graph.Weighted_Adjacency(A.tolist(), mode="undirected", loops=False)
    for e in g.es:
        e["weight"] = float(W[e.source, e.target])
    pyr.seed(seed)
    np.random.seed(seed)
    if mode == "drl":
        lay = g.layout_drl(weights="weight")
    else:
        lay = g.layout_fruchterman_reingold(weights="weight", niter=niter)
    return np.array(lay.coords)


def backbone_edges(W, labels=None, k=3):
    """The k strongest edges per node — a readable 'backbone' instead of the
    full O(N^2) edge set (which paints a hairball).

    W      : symmetric (N,N) coupling matrix.
    labels : optional per-node cluster array; nodes with label < 0 (tail /
             uncoupled) are skipped and never linked.
    k      : keep each node's k heaviest couplings.

    Returns list of (a, b, w) with a < b, deduplicated — pass straight to a
    line-drawing loop with linewidth scaled by w.
    """
    W = np.asarray(W)
    N = W.shape[0]
    lab = None if labels is None else np.asarray(labels)
    E = []
    for a in range(N):
        if lab is not None and lab[a] < 0:
            continue
        row = W[a].copy()
        row[a] = 0.0
        top = np.argsort(row)[::-1][:k]
        for b in top:
            if row[b] > 0 and a < b and (lab is None or lab[b] >= 0):
                E.append((a, int(b), float(row[b])))
    return E
