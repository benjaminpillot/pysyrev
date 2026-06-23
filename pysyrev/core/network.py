from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
import pandas as pd

from pysyrev.core.bib import ID, TITLE, YEAR, JOURNAL, DOI

_SEP = '; '
_NODE_ATTRS = [TITLE, YEAR, JOURNAL, DOI]


def _build_doc_refs(
    df:             pd.DataFrame,
    use_resolved:   bool,
    use_unresolved: bool,
) -> dict[str, set[str]]:
    """Return {doc_id: set of reference keys} for every row in df.

    Reference keys are prefixed with 'R:' (resolved internal IDs) or 'U:'
    (unresolved raw strings) to avoid accidental collisions.
    """
    doc_refs: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        refs: set[str] = set()
        if use_resolved and isinstance(row.get('reference_ids'), str):
            refs.update(f"R:{r}" for r in row['reference_ids'].split(_SEP))
        if use_unresolved and isinstance(row.get('unresolved_references'), str):
            refs.update(f"U:{r}" for r in row['unresolved_references'].split(_SEP))
        doc_refs[row[ID]] = refs
    return doc_refs


def build_coupling_graph(
    df:             pd.DataFrame,
    use_resolved:   bool = True,
    use_unresolved: bool = True,
    min_shared:     int  = 1,
) -> nx.Graph:
    """Build a bibliographic coupling graph from a bib DataFrame.

    Two documents are linked when they share at least *min_shared* references.

    Nodes  — internal document IDs, with title/year/journal/doi as attributes.
    Edges  — pairs of documents sharing ≥ min_shared references.
             ``weight``      : number of shared references.
             ``shared_refs`` : set of shared reference keys.

    Parameters
    ----------
    df : pd.DataFrame
        BibDataset internal DataFrame. Must have an 'id' column and at least
        one of 'reference_ids' / 'unresolved_references'.
    use_resolved : bool
        Include resolved internal reference IDs (column 'reference_ids').
    use_unresolved : bool
        Include unresolved raw reference strings (column 'unresolved_references').
    min_shared : int
        Minimum number of shared references to add an edge.
    """
    if not use_resolved and not use_unresolved:
        raise ValueError(
            "At least one of use_resolved or use_unresolved must be True."
        )

    doc_refs = _build_doc_refs(df, use_resolved, use_unresolved)

    # Inverted index: ref_key → list of doc_ids that cite it
    ref_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id, refs in doc_refs.items():
        for ref in refs:
            ref_to_docs[ref].append(doc_id)

    # Accumulate shared reference keys per (doc_i, doc_j) pair
    edge_shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ref, doc_ids in ref_to_docs.items():
        if len(doc_ids) < 2:
            continue
        for u, v in combinations(doc_ids, 2):
            edge_shared[(u, v)].add(ref)

    G = nx.Graph()

    for _, row in df.iterrows():
        attrs = {col: row[col] for col in _NODE_ATTRS if col in df.columns}
        G.add_node(row[ID], **attrs)

    for (u, v), shared in edge_shared.items():
        if len(shared) >= min_shared:
            G.add_edge(u, v, weight=len(shared), shared_refs=shared)

    return G


def build_citation_graph(
    df:              pd.DataFrame,
    use_resolved:    bool = True,
    use_unresolved:  bool = True,
) -> nx.DiGraph:
    """Build a directed citation graph from a bib DataFrame.

    Nodes  — corpus document IDs (node_type='internal', with title/year/journal/doi)
             plus external references (node_type='external', prefixed with R: or U:).
    Edges  — directed A → B meaning corpus document A cites reference B.

    Parameters
    ----------
    df : pd.DataFrame
        BibDataset internal DataFrame.
    use_resolved : bool
        Include resolved internal reference IDs (column 'reference_ids').
    use_unresolved : bool
        Include unresolved raw reference strings (column 'unresolved_references').
    """
    if not use_resolved and not use_unresolved:
        raise ValueError(
            "At least one of use_resolved or use_unresolved must be True."
        )

    doc_refs     = _build_doc_refs(df, use_resolved, use_unresolved)
    corpus_index = df.set_index(ID)

    G = nx.DiGraph()

    # Add all corpus docs as nodes
    for _, row in df.iterrows():
        attrs = {col: row[col] for col in _NODE_ATTRS if col in df.columns}
        attrs['node_type'] = 'internal'
        G.add_node(row[ID], **attrs)

    # Add directed edges: corpus doc → cited reference
    for doc_id, refs in doc_refs.items():
        for key in refs:
            if key.startswith('R:'):
                ref_id = key[2:]
                if ref_id in corpus_index.index:
                    # Corpus → corpus: target node already exists (no prefix)
                    G.add_edge(doc_id, ref_id)
                else:
                    # Corpus → resolved-but-external
                    if key not in G:
                        G.add_node(key, node_type='external')
                    G.add_edge(doc_id, key)
            else:
                # Corpus → unresolved external reference
                if key not in G:
                    G.add_node(key, label=key[2:], node_type='external')
                G.add_edge(doc_id, key)

    return G


def build_cocitation_graph(
    df:              pd.DataFrame,
    use_resolved:    bool = True,
    use_unresolved:  bool = True,
    min_cocitations: int  = 1,
) -> nx.Graph:
    """Build a co-citation graph from a bib DataFrame.

    Two references are co-cited when they appear together in the reference list
    of at least *min_cocitations* documents in the corpus.

    Nodes  — reference keys (R: prefix for resolved internal IDs, U: for
             unresolved raw strings). Resolved nodes whose ID exists in the
             corpus carry title/year/journal/doi attributes and
             ``node_type='internal'``; unresolved nodes have
             ``node_type='external'`` and a ``label`` attribute with the raw
             reference string.
    Edges  — pairs of references co-cited by ≥ min_cocitations documents.
             ``weight`` : co-citation count.

    Parameters
    ----------
    df : pd.DataFrame
        BibDataset internal DataFrame.
    use_resolved : bool
        Include resolved internal reference IDs (column 'reference_ids').
    use_unresolved : bool
        Include unresolved raw reference strings (column 'unresolved_references').
    min_cocitations : int
        Minimum co-citation count to add an edge.
    """
    if not use_resolved and not use_unresolved:
        raise ValueError(
            "At least one of use_resolved or use_unresolved must be True."
        )

    doc_refs = _build_doc_refs(df, use_resolved, use_unresolved)

    # Co-citation counter: (ref_a, ref_b) → number of documents citing both
    cocit_count: Counter[tuple[str, str]] = Counter()
    for refs in doc_refs.values():
        for pair in combinations(sorted(refs), 2):
            cocit_count[pair] += 1

    G = nx.Graph()

    # Node attributes: resolved refs that exist in the corpus get full metadata
    corpus_index = df.set_index(ID)
    all_ref_keys = {key for refs in doc_refs.values() for key in refs}

    for key in all_ref_keys:
        if key.startswith('R:'):
            internal_id = key[2:]
            if internal_id in corpus_index.index:
                row = corpus_index.loc[internal_id]
                attrs = {col: row[col] for col in _NODE_ATTRS if col in df.columns}
                attrs['node_type'] = 'internal'
            else:
                attrs = {'node_type': 'internal'}
            G.add_node(key, **attrs)
        else:
            G.add_node(key, label=key[2:], node_type='external')

    for (a, b), count in cocit_count.items():
        if count >= min_cocitations:
            G.add_edge(a, b, weight=count)

    return G
