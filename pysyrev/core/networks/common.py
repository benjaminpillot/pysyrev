"""
Shared analysis toolkit for the reworked bibliographic networks.

Every reworked network (coupling, co-citation, and — partly — citation) is built
the same way: a weighted similarity matrix ``W`` over nodes, then Leiden
communities, a force-directed layout, and a readable edge backbone, all driven
by the SAME ``W`` so community structure and spatial structure agree. Only the
matrix builder differs per network type (e.g. :func:`~pysyrev.core.networks.coupling.salton_coupling`).

This module holds the type-agnostic *analysis* pieces (rendering lives in
:mod:`.plotting`):

  * :func:`reference_sets`     — each document's set of cited references.
  * :class:`NetworkResult`     — the (node_ids, W, labels, coords, modularity) bundle.
  * :func:`leiden_communities` — modularity communities on any weighted ``W``.
  * :func:`force_layout`       — weighted Fruchterman-Reingold / DRL layout on ``W``.
  * :func:`backbone_edges`     — each node's ``k`` strongest edges (kills the hairball).
  * :func:`cluster_terms`      — distinguishing TF-IDF terms per community.
  * :func:`community_topic_crosstab` — community × topic contingency (the croisement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pysyrev.core.bib import ID, REFS

_SEP = "; "


# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------

def reference_sets(df, ref_col: str = REFS,
                   sep: str = _SEP) -> Tuple[List, List[set]]:
    """Return ``(node_ids, refsets)`` from the raw references column.

    Each reference token is used verbatim as an identity key — exact overlap
    only, no fuzzy resolution (this is why OpenAlex ``referenced_works`` IDs are
    the most reliable coupling basis). Rows with no references get an empty set
    and end up uncoupled.

    Parameters
    ----------
    df : pd.DataFrame
        Bib DataFrame with an ``id`` column and *ref_col*.
    ref_col : str
        Column holding ``sep``-joined reference identifiers.
    sep : str
        Separator between references within a cell.
    """
    node_ids = list(df[ID])
    refsets: List[set] = []
    for val in df[ref_col]:
        if isinstance(val, str):
            refsets.append({r.strip() for r in val.split(sep) if r.strip()})
        else:
            refsets.append(set())
    return node_ids, refsets


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------

@dataclass
class NetworkResult:
    """Everything a matrix-based network build produces, ready to inspect/plot.

    Row *i* of ``W``, ``labels``, and ``coords`` all describe ``node_ids[i]``.
    """
    node_ids:   List
    W:          np.ndarray
    labels:     np.ndarray
    coords:     np.ndarray
    modularity: float
    terms:      dict = field(default_factory=dict)   # {community: [top terms]}

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_communities(self) -> int:
        return len({c for c in self.labels.tolist() if c >= 0})

    def community_sizes(self) -> dict:
        vals, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def plot(self, **kwargs):
        """Draw this network — see :func:`pysyrev.core.networks.plotting.plot_network`."""
        # Local import: plotting imports from this module (backbone_edges), so a
        # top-level import here would be circular.
        from pysyrev.core.networks.plotting import plot_network
        return plot_network(self.coords, self.labels, self.W, **kwargs)


# ---------------------------------------------------------------------------
# Leiden communities
# ---------------------------------------------------------------------------

def leiden_communities(W: np.ndarray, resolution: float = 0.5,
                       seed: int = 7, min_size: int = 5
                       ) -> Tuple[np.ndarray, float]:
    """Leiden communities from a weighted similarity matrix.

    Uses ``leidenalg.RBConfigurationVertexPartition`` (modularity with a
    resolution knob) on the weighted graph of ``W``'s non-zero entries — the
    same object :func:`force_layout` lays out, so communities and space agree.

    Parameters
    ----------
    W : np.ndarray
        Symmetric weighted matrix (zero diagonal).
    resolution : float
        Higher → more, smaller communities. Sweep 0.4–0.9 and report modularity.
    seed : int
        RNG seed for a reproducible partition.
    min_size : int
        Communities smaller than this are relabelled ``-1`` (the unstructured
        tail); keep them in the corpus but out of the layout.

    Returns
    -------
    (labels, modularity)
        ``labels`` is an int array row-aligned to *W* (``-1`` = tail);
        ``modularity`` is the partition's modularity score.
    """
    import igraph as ig
    import leidenalg as la

    n = W.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = [W[i, j] for i, j in edges]

    part = la.find_partition(
        g, la.RBConfigurationVertexPartition,
        weights=g.es["weight"] if edges else None,
        resolution_parameter=resolution, seed=seed,
    )
    labels = np.array(part.membership)
    sizes = {c: int((labels == c).sum()) for c in set(labels.tolist())}
    labels = np.array([c if sizes[c] >= min_size else -1 for c in labels])
    return labels, float(part.modularity)


# ---------------------------------------------------------------------------
# Layout + backbone
# ---------------------------------------------------------------------------

def force_layout(W: np.ndarray, seed: int = 7, niter: int = 500,
                 mode: str = "fr") -> np.ndarray:
    """2-D coordinates laid out so strongly-linked nodes sit together.

    A weighted Fruchterman-Reingold (``mode="fr"``) or DRL (``mode="drl"``,
    scales past a few thousand nodes) layout on ``W``. Because the edge weights
    ARE the similarity values, nodes Leiden grouped attract spatially too.

    Returns ``(N, 2)`` coordinates, row-aligned to *W*. Isolated nodes drift to
    the rim.
    """
    import random as pyr
    import igraph as ig

    A = (np.asarray(W) > 0).astype(float)
    g = ig.Graph.Weighted_Adjacency(A.tolist(), mode="undirected", loops=False)
    for e in g.es:
        e["weight"] = float(W[e.source, e.target])
    # Seed the RNGs beforehand — igraph's layout call rejects a seed= kwarg.
    pyr.seed(seed)
    np.random.seed(seed)
    if mode == "drl":
        lay = g.layout_drl(weights="weight")
    else:
        lay = g.layout_fruchterman_reingold(weights="weight", niter=niter)
    return np.array(lay.coords)


def backbone_edges(W: np.ndarray, labels: Optional[np.ndarray] = None,
                   k: int = 3) -> List[Tuple[int, int, float]]:
    """The ``k`` strongest edges per node — a readable backbone.

    Instead of the full ``O(N^2)`` edge set (a hairball), keep each node's ``k``
    heaviest links. Nodes with ``labels < 0`` (tail) are skipped.

    Returns a deduplicated list of ``(a, b, w)`` with ``a < b`` — pass straight
    to a line-drawing loop with linewidth scaled by ``w``.
    """
    W = np.asarray(W)
    N = W.shape[0]
    lab = None if labels is None else np.asarray(labels)
    E: List[Tuple[int, int, float]] = []
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


# ---------------------------------------------------------------------------
# Cluster labelling & croisement (optional)
# ---------------------------------------------------------------------------

def cluster_terms(texts: Sequence[str], labels: np.ndarray, top: int = 12,
                  ngram: Tuple[int, int] = (1, 2), min_df: int = 3,
                  max_df: float = 0.4) -> dict:
    """Distinguishing TF-IDF terms per community (mean tf-idf ranking).

    *texts* must be row-aligned to *labels*. The ``-1`` tail is skipped.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels = np.asarray(labels)
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram,
                          min_df=min_df, max_df=max_df)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        m = labels == c
        mean = np.asarray(X[m].mean(axis=0)).ravel()
        out[c] = vocab[np.argsort(-mean)[:top]].tolist()
    return out


def community_topic_crosstab(labels, topics, topic_labels: Optional[dict] = None):
    """Contingency table of Leiden community (rows) × topic (columns).

    Row-aligned *labels* and *topics* arrays. Quantifies the croisement: how the
    documents of each bibliographic community distribute across textual topics.
    Returns a pandas DataFrame of counts.
    """
    import pandas as pd

    df = pd.DataFrame({"community": np.asarray(labels), "topic": np.asarray(topics)})
    ct = pd.crosstab(df["community"], df["topic"])
    if topic_labels:
        ct = ct.rename(columns={t: topic_labels.get(t, t) for t in ct.columns})
    return ct
