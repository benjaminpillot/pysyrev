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

Unlike coupling, whose nodes are corpus documents carrying their own title and
abstract, a co-citation node is usually an extra-corpus work the corpus knows
nothing about. Pass *ref_meta* (from
:func:`pysyrev.core.references.fetch_reference_metadata`) to give those nodes
back their content: the build then derives per-community TF-IDF sub-themes just
like coupling does, and :func:`cluster_profiles` summarises each community's
authors, venues and period — the "intellectual base" read of a cluster.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pysyrev.core.bib import REFS
from pysyrev.core.networks.common import (
    NetworkResult,
    cluster_terms,
    force_layout,
    leiden_best_resolution,
    leiden_communities,
    reference_sets,
    resolution_grid,
)


def frequent_references(refsets: Sequence[set],
                        min_ref_freq: int = 2) -> Tuple[List, Counter]:
    """``(kept, freq)`` — the references cited by at least *min_ref_freq*
    documents, and the citation frequency of every reference.

    The co-citation node set, derived from the reference sets alone. Split out of
    :func:`cocitation_matrix` because a caller may need to know the nodes before
    paying for the matrix — resolving their metadata, say.
    """
    freq: Counter = Counter()
    for refs in refsets:
        freq.update(refs)
    return [r for r, c in freq.items() if c >= min_ref_freq], freq


def cocitation_matrix(refsets: Sequence[set],
                      min_ref_freq: int = 2) -> Tuple[List, np.ndarray]:
    """Salton-normalised co-citation matrix over references.

    ``W[a,b] = cocited(a,b) / sqrt(freq(a) * freq(b))`` where ``cocited`` is the
    number of documents citing both references and ``freq`` is how many
    documents cite a reference. Only references cited by at least
    *min_ref_freq* documents are kept as nodes.

    Returns ``(ref_ids, W)`` — ``ref_ids[i]`` labels row ``i`` of ``W``.
    """
    kept, freq = frequent_references(refsets, min_ref_freq=min_ref_freq)
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


def reference_texts(node_ids: Sequence, ref_meta: Dict[str, dict]) -> List[str]:
    """One text per node, row-aligned to *node_ids*, for TF-IDF.

    Concatenates each reference's title, venue and (when it was fetched) abstract.
    Nodes with no metadata yield an empty string: they contribute nothing to
    their community's terms rather than breaking the row alignment.
    """
    texts = []
    for nid in node_ids:
        entry = ref_meta.get(nid) or {}
        parts = [entry.get("title"), entry.get("venue"), entry.get("abstract")]
        texts.append(". ".join(p for p in parts if p))
    return texts


def _reference_subthemes(node_ids, labels, ref_meta: Dict[str, dict],
                         min_df: int = 2, max_df: float = 0.7) -> dict:
    """TF-IDF sub-theme terms per Leiden community, from the references' own text.

    The co-citation counterpart of :func:`~.coupling._cluster_subthemes`, on the
    resolved node metadata instead of the corpus text. The vectoriser is looser
    than that function's: a reference contributes a title (plus a venue, and an
    abstract only when one was fetched) where a coupling node contributes a full
    title + abstract, so :func:`cluster_terms`'s defaults — a term in at least 3
    nodes and at most 40% of them — leave nothing behind.

    Guarded the same way: too little text, or any vectoriser hiccup, yields ``{}``
    rather than raising, so the network still builds.
    """
    if not ref_meta:
        return {}
    texts = reference_texts(node_ids, ref_meta)
    if sum(1 for t in texts if t.strip()) < 3:
        return {}
    try:
        return cluster_terms(texts, labels, min_df=min_df, max_df=max_df)
    except Exception:             # empty vocabulary / any vectoriser hiccup
        return {}                 # sub-themes are optional; never break the network


def cluster_profiles(node_ids: Sequence, labels, W, ref_meta: Dict[str, dict],
                     top_authors: int = 8, top_venues: int = 3,
                     top_works: int = 5) -> Dict[int, dict]:
    """Per-community profile of a co-citation partition.

    For each Leiden community (the ``-1`` tail is skipped), summarises what its
    references actually are: the most frequently appearing authors and venues,
    the period they span, and the works with the highest co-citation strength.

    Returns ``{community: {n_refs, n_known, authors, venues, median_year,
    year_range, works}}``, where ``authors``/``venues`` are ``(name, count)``
    pairs and ``works`` holds the top metadata dicts with the node's ``key`` and
    ``strength`` added. Communities where no node carries metadata are omitted;
    an empty *ref_meta* therefore yields ``{}``.
    """
    labels = np.asarray(labels)
    strength = np.asarray(W).sum(axis=1)
    out: Dict[int, dict] = {}

    for comm in sorted({int(c) for c in labels.tolist() if c >= 0}):
        idx = np.where(labels == comm)[0]
        known = [(i, ref_meta.get(node_ids[i])) for i in idx]
        known = [(i, m) for i, m in known if m and m.get("title")]
        if not known:
            continue

        authors: Counter = Counter()
        venues: Counter = Counter()
        years: List[int] = []
        for _, m in known:
            authors.update(m.get("authors") or [])
            if m.get("venue"):
                venues[m["venue"]] += 1
            if m.get("year"):
                years.append(int(m["year"]))

        ranked = sorted(known, key=lambda pair: strength[pair[0]], reverse=True)
        out[comm] = {
            "n_refs":      int(len(idx)),
            "n_known":     len(known),
            "authors":     authors.most_common(top_authors),
            "venues":      venues.most_common(top_venues),
            "median_year": int(np.median(years)) if years else None,
            "year_range":  (min(years), max(years)) if years else None,
            "works":       [dict(m, key=node_ids[i], strength=float(strength[i]))
                            for i, m in ranked[:top_works]],
        }
    return out


def build_cocitation(df, ref_col: str = REFS, min_ref_freq: int = 2,
                     resolution: float = 0.7, seed: int = 7, min_size: int = 5,
                     layout_niter: int = 500,
                     resolution_range: Optional[Sequence[float]] = None,
                     resolution_step: float = 0.1,
                     ref_meta: Optional[Dict[str, dict]] = None) -> NetworkResult:
    """Build the full co-citation network from a bib DataFrame.

    Pipeline: reference sets → co-citation matrix → Leiden communities → force
    layout → (with *ref_meta*) per-community TF-IDF sub-themes. Returns a
    :class:`NetworkResult` whose ``node_ids`` are reference identifiers (not
    corpus documents). Returns an empty result when fewer than two references
    clear *min_ref_freq*.

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
    ref_meta : {reference_key: metadata}, optional
        Resolved metadata for the reference nodes, from
        :func:`pysyrev.core.references.fetch_reference_metadata`. Only the nodes
        it covers are enriched; supplying it fills ``terms`` (TF-IDF sub-themes)
        and ``node_meta`` on the result, which is what makes a co-citation
        community nameable. Without it the network builds exactly as before.
    """
    ref_meta = ref_meta or {}
    _, refsets = reference_sets(df, ref_col=ref_col)
    ref_ids, W = cocitation_matrix(refsets, min_ref_freq=min_ref_freq)

    if len(ref_ids) < 2:
        return NetworkResult(node_ids=ref_ids, W=W,
                             labels=np.full(len(ref_ids), -1, dtype=int),
                             coords=np.zeros((len(ref_ids), 2)), modularity=0.0,
                             node_meta={k: v for k, v in ref_meta.items()
                                        if k in set(ref_ids)})

    if resolution_range is not None:
        labels, modularity, resolution, sweep = leiden_best_resolution(
            W, resolution_grid(resolution_range, resolution_step),
            seed=seed, min_size=min_size)
    else:
        labels, modularity = leiden_communities(
            W, resolution=resolution, seed=seed, min_size=min_size)
        sweep = None
    coords = force_layout(W, seed=seed, niter=layout_niter)
    terms = _reference_subthemes(ref_ids, labels, ref_meta)
    return NetworkResult(node_ids=ref_ids, W=W, labels=labels,
                         coords=coords, modularity=modularity,
                         resolution=resolution, resolution_sweep=sweep,
                         terms=terms,
                         node_meta={k: v for k, v in ref_meta.items()
                                    if k in set(ref_ids)})
