---
name: cluster-connectivity-matrix
description: Quantify how bibliographic-coupling clusters connect to each other. Companion to corpus-map. Builds the inter-cluster mean-coupling matrix from a symmetric coupling matrix W and a per-node cluster labelling, then reports each cluster's inward (internal cohesion) vs outward (coupling to the rest of the corpus) profile against the corpus-wide baseline. Use to test whether a cluster is self-contained / weakly integrated, and to feed a connectivity-matrix heatmap and inward/outward bar panels. Handles split sub-clusters (pass explicit (label, idx) groups).
---

# Cluster connectivity matrix — inward/outward coupling

After clustering a corpus (see `corpus-map`), this measures **how the clusters
relate to each other**, not just their internal structure. The unit throughout
is the mean **bibliographic-coupling** weight between two sets of papers — the
same `W` that clustering ran on.

## Inputs (already in the kernel after corpus-map)

- `W` — symmetric `(N,N)` coupling matrix (Salton-normalised shared references).
- a per-node integer `labels` array, or explicit `(label, idx_array)` groups.

## What the numbers mean

- **Internal cohesion** (matrix diagonal): mean coupling over every distinct
  *intra*-cluster pair — how tightly a cluster's own papers share references.
  Computed on the strict upper triangle so the zero self-diagonal never dilutes it.
- **Outward coupling** (off-diagonal row mean): how much a cluster shares
  references with the *rest* of the corpus.
- **Corpus baseline**: mean coupling over all distinct pairs — the reference to
  judge any block as high or low.

A cluster with internal ≫ outward, and outward well below baseline, is
**self-contained / weakly integrated** — a candidate emerging or siloed theme.

## Workflow

```python
from collections import OrderedDict
import numpy as np

# 1. groups: order defines the matrix order. Simplest from a labels array:
groups = [(f"C{c}", np.where(labels == c)[0]) for c in sorted(set(labels)) if c >= 0]

# 1b. split a cluster into sub-clusters? pass explicit member positions, e.g.
#     groups = [("C0", idxs[0]), ("C2-sub0", sub0_pos), ("C2-sub1", sub1_pos), ...]

# 2. the matrix + baseline
M, labs = inter_cluster_matrix(W, groups, scale=1000)
base = corpus_baseline(W, scale=1000)

# 3. per-cluster inward/outward
for i, lab in enumerate(labs):
    p = coupling_inout(M, i)
    print(f"{lab}: internal {p['internal']:.2f} | outward {p['outward']:.2f} | baseline {base:.2f}")
```

## Figure (heatmap + inward/outward bars)

Render `M` as an imshow heatmap (annotate each cell, mask/redden near-zero
off-diagonal rows to spotlight an isolated cluster), and a grouped bar panel of
`internal` vs `outward` per cluster with the `baseline` drawn as a dashed
`axhline`. Load `figure-style` and call `apply_figure_style()` first; keep the
colorbar (`fraction=0.046, pad=0.04`) clear of the last matrix column with
generous `wspace`.

## Notes

- `scale=1000` only rescales for readability; ratios and the baseline comparison
  are unaffected.
- `M` is symmetric — `M[i,j]` is undirected shared-reference overlap, not a
  directed citation. For *directed* citation flow between clusters, count
  `referenced_works` edges instead (a different measure).
- Tail nodes (`labels == -1`, uncoupled) are excluded by the `c >= 0` guard.
