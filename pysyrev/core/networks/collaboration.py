"""
Who drives a literature — actor rankings and collaboration networks.

Built only on the canonical :data:`~pysyrev.core.actors.ACTORS` column, so it
works the same whatever source(s) the corpus came from. One *level* at a time —
``"author"``, ``"institution"`` or ``"country"``:

  * :func:`actor_frame`          — the long (paper × actor) table everything reads.
  * :func:`rank_actors`          — most involved actors: papers (full and
                                   fractional counting), citations, active years.
  * :func:`collaboration_matrix` — actor × actor co-occurrence on shared papers.
  * :func:`build_collaboration`  — that matrix as a :class:`NetworkResult`
                                   (Leiden communities + force layout), the same
                                   toolkit as the coupling network.
  * :func:`collaboration_summary`— coverage and collaboration indicators
                                   (team size, multi-institution, international).

Counting: *full* counting credits each actor 1 per paper it appears on;
*fractional* counting splits each paper's unit among its actors of that level
(1/n), so a 40-country consortium paper does not outweigh 40 national ones.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from pysyrev.core.actors import ACTORS, LEVELS, level_entities
from pysyrev.core.bib import CITED_BY, ID, YEAR
from pysyrev.core.networks.common import (
    NetworkResult,
    force_layout,
    leiden_best_resolution,
    leiden_communities,
    resolution_grid,
)


def _check_level(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"unknown actor level {level!r}; expected one of {LEVELS}")


def actor_frame(df: pd.DataFrame, level: str, col: str = ACTORS) -> pd.DataFrame:
    """Long table of (paper, actor) pairs for one *level*.

    Columns: ``doc`` (row position in *df*), ``id``, ``year``, ``cited_by``,
    ``key``, ``name``, ``n_level`` (how many actors of that level the paper
    has — the fractional-counting denominator) and, for institutions,
    ``country``. Papers without the column or without actors at that level
    contribute no row.
    """
    _check_level(level)
    columns = ["doc", "id", "year", "cited_by", "key", "name", "n_level", "country"]
    if col not in df.columns:
        return pd.DataFrame(columns=columns)
    ids = df[ID].tolist() if ID in df.columns else list(range(len(df)))
    years = df[YEAR].tolist() if YEAR in df.columns else [None] * len(df)
    cites = df[CITED_BY].tolist() if CITED_BY in df.columns else [None] * len(df)
    rows = []
    for doc, (cell, pid, yr, cb) in enumerate(zip(df[col], ids, years, cites)):
        ents = level_entities(cell, level)
        for e in ents:
            rows.append((doc, pid, yr, cb, e["key"], e.get("name"), len(ents),
                         e.get("country") if level == "institution" else None))
    out = pd.DataFrame(rows, columns=columns)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["cited_by"] = pd.to_numeric(out["cited_by"], errors="coerce").fillna(0)
    return out


def _mode(values) -> Optional[str]:
    values = [v for v in values if isinstance(v, str) and v]
    return Counter(values).most_common(1)[0][0] if values else None


def actor_names(long: pd.DataFrame) -> Dict[str, str]:
    """Display name of each key: its most frequent spelling in the corpus."""
    return {k: _mode(g) or k for k, g in long.groupby("key")["name"]}


def rank_actors(df: pd.DataFrame, level: str, top: Optional[int] = 20,
                sort_by: str = "papers", min_papers: int = 1,
                col: str = ACTORS) -> pd.DataFrame:
    """The actors most involved in the corpus at one *level*.

    Returns one row per actor, sorted by *sort_by* (``"papers"``,
    ``"fractional"`` or ``"citations"``, ties broken by the others):

    ``key``, ``name``, ``papers`` (full count), ``fractional`` (1/n count),
    ``share`` (papers / corpus papers carrying that level), ``citations``
    (summed over its papers), ``citations_per_paper``, ``first_year``,
    ``last_year`` and, for institutions, ``country`` (its most frequent one).
    *top* ``None`` keeps every actor with at least *min_papers* papers.
    """
    order = {"papers": ["papers", "fractional", "citations"],
             "fractional": ["fractional", "papers", "citations"],
             "citations": ["citations", "papers", "fractional"]}
    if sort_by not in order:
        raise ValueError(f"sort_by must be one of {list(order)}, got {sort_by!r}")
    long = actor_frame(df, level, col=col)
    cols = ["key", "name", "papers", "fractional", "share", "citations",
            "citations_per_paper", "first_year", "last_year"]
    if level == "institution":
        cols.append("country")
    if long.empty:
        return pd.DataFrame(columns=cols)
    long = long.assign(frac=1.0 / long["n_level"])
    g = long.groupby("key")
    out = pd.DataFrame({
        "papers":     g["doc"].nunique(),
        "fractional": g["frac"].sum(),
        "citations":  g["cited_by"].sum(),
        "first_year": g["year"].min(),
        "last_year":  g["year"].max(),
    })
    names = actor_names(long)
    out["name"] = [names[k] for k in out.index]
    n_docs = long["doc"].nunique()
    out["share"] = out["papers"] / n_docs if n_docs else np.nan
    out["citations_per_paper"] = out["citations"] / out["papers"]
    if level == "institution":
        out["country"] = g["country"].agg(_mode)
    out = (out[out["papers"] >= min_papers]
           .sort_values(order[sort_by], ascending=False)
           .reset_index().rename(columns={"index": "key"}))
    out = out[cols]
    for c in ("first_year", "last_year"):
        out[c] = out[c].round().astype("Int64")
    return out.head(top) if top else out


def collaboration_matrix(df: pd.DataFrame, level: str, min_papers: int = 2,
                         max_nodes: Optional[int] = 300,
                         normalize: str = "salton", col: str = ACTORS):
    """Actor × actor collaboration matrix: papers shared by each pair.

    Nodes are the actors with at least *min_papers* papers, capped to the
    *max_nodes* most productive (the matrix is dense). *normalize*:

    ``"count"``        raw number of co-authored papers;
    ``"salton"``       ``c_ij / sqrt(p_i p_j)`` — the coupling network's
                       normalisation, so prolific actors do not link to all;
    ``"association"``  ``c_ij / (p_i p_j)`` rescaled to mean 1 (VOSviewer's
                       association strength): observed over expected.

    Returns ``(keys, W, papers)`` where ``papers[i]`` is node *i*'s paper count.
    """
    if normalize not in ("count", "salton", "association"):
        raise ValueError(f"normalize must be count|salton|association, got {normalize!r}")
    long = actor_frame(df, level, col=col)
    if long.empty:
        return [], np.zeros((0, 0)), np.zeros(0)
    papers = long.groupby("key")["doc"].nunique()
    papers = papers[papers >= min_papers].sort_values(ascending=False)
    if max_nodes:
        papers = papers.head(max_nodes)
    keys = papers.index.tolist()
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    C = np.zeros((n, n))
    for _, g in long[long["key"].isin(pos)].groupby("doc"):
        idx = sorted({pos[k] for k in g["key"]})
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                C[idx[a], idx[b]] += 1
                C[idx[b], idx[a]] += 1
    p = papers.to_numpy(dtype=float)
    if normalize == "salton":
        C = C / np.sqrt(np.outer(p, p))
    elif normalize == "association":
        C = C / np.outer(p, p)
        nz = C[C > 0]
        if nz.size:
            C = C / nz.mean()
    np.fill_diagonal(C, 0.0)
    return keys, C, p


def build_collaboration(df: pd.DataFrame, level: str, min_papers: int = 2,
                        max_nodes: Optional[int] = 300, normalize: str = "salton",
                        resolution: float = 1.0,
                        resolution_range: Optional[Sequence[float]] = None,
                        resolution_step: float = 0.1, min_size: int = 3,
                        seed: int = 7, layout_niter: int = 500,
                        col: str = ACTORS) -> NetworkResult:
    """Collaboration network of one *level*, as a :class:`NetworkResult`.

    :func:`collaboration_matrix` → Leiden communities (fixed *resolution*, or
    the best of a *resolution_range* sweep) → force layout on the same ``W``.
    ``node_meta[key]`` carries ``name``, ``papers``, ``fractional``,
    ``citations`` (and ``country`` for institutions), so the plot hover and
    the report tables need nothing else.
    """
    keys, W, _ = collaboration_matrix(df, level, min_papers=min_papers,
                                      max_nodes=max_nodes, normalize=normalize,
                                      col=col)
    if len(keys) < 2:
        raise ValueError(f"{level} collaboration network needs at least 2 actors "
                         f"with >= {min_papers} papers (found {len(keys)})")
    if resolution_range is not None:
        labels, modularity, resolution, sweep = leiden_best_resolution(
            W, resolution_grid(resolution_range, resolution_step),
            seed=seed, min_size=min_size)
    else:
        labels, modularity = leiden_communities(W, resolution=resolution,
                                                seed=seed, min_size=min_size)
        sweep = None
    coords = force_layout(W, seed=seed, niter=layout_niter)
    ranking = rank_actors(df, level, top=None, col=col).set_index("key")
    meta_cols = [c for c in ("name", "papers", "fractional", "citations", "country")
                 if c in ranking.columns]
    node_meta = {k: ranking.loc[k, meta_cols].to_dict() for k in keys if k in ranking.index}
    return NetworkResult(node_ids=keys, W=W, labels=labels, coords=coords,
                         modularity=modularity, node_meta=node_meta,
                         resolution=resolution, resolution_sweep=sweep)


def collaboration_summary(df: pd.DataFrame, col: str = ACTORS) -> dict:
    """Coverage and collaboration indicators of the corpus.

    ``n_papers``; per level, ``with_<level>`` (papers carrying at least one
    actor of it — read the other figures against it) and ``n_<level>``
    (distinct actors); ``mean_authors``; ``single_author_share``;
    ``multi_institution_share`` and ``international_share`` (papers with >= 2
    institutions / countries, among those with any).
    """
    out = {"n_papers": int(len(df))}
    counts = {}
    for level in LEVELS:
        long = actor_frame(df, level, col=col)
        per_doc = long.groupby("doc")["key"].nunique()
        counts[level] = per_doc
        out[f"with_{level}"] = int(per_doc.size)
        out[f"n_{level}"] = int(long["key"].nunique()) if not long.empty else 0

    def share(s, k):
        return float((s >= k).mean()) if s.size else float("nan")

    a = counts["author"]
    out["mean_authors"] = float(a.mean()) if a.size else float("nan")
    out["single_author_share"] = float((a == 1).mean()) if a.size else float("nan")
    out["multi_institution_share"] = share(counts["institution"], 2)
    out["international_share"] = share(counts["country"], 2)
    return out
