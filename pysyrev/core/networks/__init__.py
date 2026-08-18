"""Reworked bibliographic networks (coupling first; co-citation/citation next).

Shared, type-agnostic machinery lives in :mod:`.common`; each network type
contributes only its similarity-matrix builder and a thin ``build_*`` compose
function.
"""

from pysyrev.core.networks.common import (
    NetworkResult,
    backbone_edges,
    cluster_terms,
    community_topic_crosstab,
    force_layout,
    leiden_communities,
    reference_sets,
)
from pysyrev.core.networks.coupling import build_coupling, salton_coupling
from pysyrev.core.networks.cocitation import build_cocitation, cocitation_matrix
from pysyrev.core.networks.plotting import plot_network

__all__ = [
    "NetworkResult",
    "reference_sets",
    "leiden_communities",
    "force_layout",
    "backbone_edges",
    "cluster_terms",
    "community_topic_crosstab",
    "plot_network",
    "salton_coupling",
    "build_coupling",
    "cocitation_matrix",
    "build_cocitation",
]
