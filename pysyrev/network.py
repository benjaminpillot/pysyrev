import networkx as nx
import pandas as pd

from pysyrev.bibdata import BibDataset
from pysyrev.core.config import BibNetworkConfig, BibNetworkExportConfig
from pysyrev.core.network import (
    build_coupling_graph,
    build_cocitation_graph,
    build_citation_graph,
)


class BibNetwork:
    """Network analysis built from a BibDataset.

    Three graph types are available:

    Citation — directed graph where A → B means corpus document A cites B.
        Nodes are corpus docs plus (optionally) external references.
        See build_citation_network().

    Bibliographic coupling — documents as nodes, linked when they share
        common references. See build_coupling_network().

    Co-citation — references as nodes, linked when they appear together
        in the reference list of at least one document. Resolved references
        that exist in the corpus are marked node_type='internal'; unresolved
        external references are node_type='external'. See build_cocitation_network().

    Usage
    -----
    net = BibNetwork(bib_dataset)
    net.build_citation_network()
    net.build_coupling_network()
    net.build_cocitation_network()
    G_citation  = net.citation_graph
    G_coupling  = net.coupling_graph
    G_cocit     = net.cocitation_graph
    """

    _citation_config   = None
    _coupling_config   = None
    _cocitation_config = None
    _export_config     = None
    _doc_dataset       = None
    _citation_graph    = None
    _coupling_graph    = None
    _cocitation_graph  = None

    def __init__(self, bib_dataset=None):
        self._dataset = bib_dataset

    # ------------------------------------------------------------------ #
    # Direct citation                                                       #
    # ------------------------------------------------------------------ #

    def build_citation_network(self) -> 'BibNetwork':
        """Build (or rebuild) the directed citation graph."""
        self._citation_graph = build_citation_graph(self._dataset.dataset)
        return self

    @property
    def citation_graph(self):
        if self._citation_graph is None:
            raise ValueError(
                "Citation graph not built yet — call build_citation_network() first."
            )
        return self._citation_graph

    # ------------------------------------------------------------------ #
    # Bibliographic coupling                                               #
    # ------------------------------------------------------------------ #

    def build_coupling_network(self, min_shared: int = 1) -> 'BibNetwork':
        """Build (or rebuild) the bibliographic coupling graph.

        Parameters
        ----------
        min_shared : int
            Minimum shared references to add an edge between two documents.
        """
        self._coupling_graph = build_coupling_graph(
            self._dataset.dataset,
            min_shared=min_shared,
        )
        return self

    @property
    def coupling_graph(self):
        if self._coupling_graph is None:
            raise ValueError(
                "Coupling graph not built yet — call build_coupling_network() first."
            )
        return self._coupling_graph

    # ------------------------------------------------------------------ #
    # Co-citation                                                          #
    # ------------------------------------------------------------------ #

    def build_cocitation_network(self, min_cocitations: int = 1) -> 'BibNetwork':
        """Build (or rebuild) the co-citation graph.

        Parameters
        ----------
        min_cocitations : int
            Minimum co-citation count to add an edge between two references.
        """
        self._cocitation_graph = build_cocitation_graph(
            self._dataset.dataset,
            min_cocitations=min_cocitations,
        )
        return self

    @property
    def cocitation_graph(self):
        if self._cocitation_graph is None:
            raise ValueError(
                "Co-citation graph not built yet — call build_cocitation_network() first."
            )
        return self._cocitation_graph

    # ------ bridge from configuration ---------------------------------

    @classmethod
    def from_config(cls, config: BibNetworkConfig) -> 'BibNetwork':
        instance = cls()
        instance._doc_dataset       = config.doc_dataset
        instance._citation_config   = config.citation_network
        instance._coupling_config   = config.coupling_network
        instance._cocitation_config = config.cocitation_network
        instance._export_config     = config.export
        return instance

    # ------ runtime ---------------------------------------------------

    def run(self, dataset: pd.DataFrame = None) -> 'BibNetwork':
        """Build citation, coupling and co-citation graphs.

        Parameters
        ----------
        dataset : pd.DataFrame, optional
            Reviewed-included dataset. If None, loaded from ``doc_dataset``
            (set via config or auto-detected by Config.load).
        """
        if dataset is not None:
            self._dataset = BibDataset(bib_dataset=dataset)
        elif self._dataset is None:
            if not self._doc_dataset:
                raise ValueError(
                    "No dataset provided: pass a DataFrame to run() or set "
                    "doc_dataset in the bib_network section of your config."
                )
            self._dataset = BibDataset(bib_dataset=pd.read_csv(self._doc_dataset))

        self.build_citation_network()
        cfg = self._coupling_config
        self.build_coupling_network(
            min_shared = cfg.min_shared if cfg else 1,
        )
        cfg = self._cocitation_config
        self.build_cocitation_network(
            min_cocitations = cfg.min_cocitations if cfg else 1,
        )
        return self

    def save(self, export_config: BibNetworkExportConfig = None) -> 'BibNetwork':
        """Export citation, coupling and co-citation graphs to GraphML files.

        GraphML is compatible with Gephi, Cytoscape, and networkx.
        ``shared_refs`` edge sets are serialised as semicolon-separated strings.
        Falls back to the export config provided at construction time (from YAML).
        """
        cfg = export_config or self._export_config
        if cfg is None:
            raise ValueError(
                "No export config: set bib_network.export in your config or "
                "pass a BibNetworkExportConfig to save()."
            )
        cfg.resolve()

        if self._citation_graph is not None:
            nx.write_graphml(self._citation_graph, cfg.citation_graph)

        if self._coupling_graph is not None:
            graph = self._coupling_graph.copy()
            for _, _, data in graph.edges(data=True):
                if isinstance(data.get('shared_refs'), set):
                    data['shared_refs'] = '; '.join(sorted(data['shared_refs']))
            nx.write_graphml(graph, cfg.coupling_graph)

        if self._cocitation_graph is not None:
            nx.write_graphml(self._cocitation_graph, cfg.cocitation_graph)

        return self

    # ------------------------------------------------------------------ #
    # Generic stats                                                        #
    # ------------------------------------------------------------------ #

    @property
    def n_citation_nodes(self) -> int:
        return self._citation_graph.number_of_nodes() if self._citation_graph is not None else 0

    @property
    def n_citation_edges(self) -> int:
        return self._citation_graph.number_of_edges() if self._citation_graph is not None else 0

    @property
    def n_coupling_nodes(self) -> int:
        return self._coupling_graph.number_of_nodes() if self._coupling_graph is not None else 0

    @property
    def n_coupling_edges(self) -> int:
        return self._coupling_graph.number_of_edges() if self._coupling_graph is not None else 0

    @property
    def n_cocitation_nodes(self) -> int:
        return self._cocitation_graph.number_of_nodes() if self._cocitation_graph is not None else 0

    @property
    def n_cocitation_edges(self) -> int:
        return self._cocitation_graph.number_of_edges() if self._cocitation_graph is not None else 0
