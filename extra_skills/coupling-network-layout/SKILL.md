---
name: coupling-network-layout
description: Lay out a bibliographic-coupling network so Leiden communities read as visually separated blobs. Companion to corpus-map. Builds a weighted Fruchterman-Reingold (or DRL) layout on the SAME coupling matrix W that clustering ran on, so force-directed attraction and modularity communities agree, then draws only a k-nearest 'backbone' of the strongest edges per node instead of the full hairball. Use for the network panel of a literature-map figure, colouring nodes by cluster and ringing seed papers. Deterministic via seed.
---

# Coupling-network layout — why the clusters look clean

The crisp, well-separated blobs in a coupling-network panel are not decorative.
They come from **two design choices that must agree**:

1. **Communities = Leiden on the coupling graph.** Clusters are
   `leidenalg.RBConfigurationVertexPartition` on the bibliographic-coupling
   matrix `W` (see `corpus-map` → `leiden_clusters`). This is a modularity
   community detection, not k-means on embeddings.
2. **Layout = force-directed on the SAME `W`.** The node positions come from a
   **weighted Fruchterman-Reingold** layout whose edge weights ARE the coupling
   values. Papers that share many references attract; since that is exactly what
   Leiden clustered on, community structure and spatial structure coincide — so
   the communities look separated. Using a layout driven by anything other than
   `W` (or clustering on something other than `W`) breaks this agreement and the
   blobs smear.

A third choice keeps it **legible**: don't draw all ~O(N²) edges. Draw only each
node's `k` strongest couplings (`backbone_edges`, k=3) — the hairball becomes a
skeleton that still traces the community boundaries.

## Workflow

```python
import numpy as np, matplotlib.pyplot as plt

# W and labels are already in the kernel from corpus-map clustering.
coords = coupling_layout(W, seed=7, niter=500)      # (N,2), row-aligned to W
E = backbone_edges(W, labels, k=3)                   # [(a,b,w), ...]

apply_figure_style()                                  # load figure-style first
fig, ax = plt.subplots(figsize=(7, 6))
for a, b, w in E:                                     # edges under nodes
    ax.plot(coords[[a, b], 0], coords[[a, b], 1],
            color="#CFCFCF", lw=0.15 + 1.2 * w, alpha=0.35, zorder=1)
for c in sorted(set(labels)):
    if c < 0:                                          # skip uncoupled tail
        continue
    m = np.where(labels == c)[0]
    ax.scatter(coords[m, 0], coords[m, 1], s=16, c=CMAP[c],
               edgecolors="white", linewidths=0.2, label=f"C{c}", zorder=3)
# ring the seed papers
sm = [i for i in range(len(labels)) if node_id[i] in seed_id_set]
ax.scatter(coords[sm, 0], coords[sm, 1], s=95, facecolors="none",
           edgecolors="#111", linewidths=1.3, zorder=4, label="seed papers")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.legend(loc="upper left", fontsize=6.5, frameon=False)
```

## Parameters & pitfalls

- **Reproducibility.** `coupling_layout` seeds both `random` and numpy. The same
  `seed` gives the same layout; change it only to escape an unlucky initialisation.
  Do NOT pass a `seed=` kwarg to igraph's layout call itself — it rejects it;
  seeding the RNGs beforehand (as the helper does) is the supported route.
- **Scale.** FR is fine to a few thousand nodes. Past that use `mode="drl"`.
- **Uncoupled nodes** (`labels == -1`) have no edges and drift to the rim; the
  `c < 0` guard keeps them out of both the backbone and the scatter.
- **k.** `k=3` is a good default; raise it for denser tracery, lower for a
  sparser skeleton. It only affects drawn edges, never the layout or clustering.
- `CMAP`, `node_id` (row→OpenAlex-id), and `seed_id_set` come from your build;
  the helper needs only `W` (+ `labels` for the backbone guard).
