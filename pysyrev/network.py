from pysyrev.core.network import build_coupling_graph, build_cocitation_graph


class BibNetwork:
    """Network analysis built from a BibDataset.

    Two graph types are available:

    Bibliographic coupling — documents as nodes, linked when they share
        common references. See build_coupling_network().

    Co-citation — references as nodes, linked when they appear together
        in the reference list of at least one document. Resolved references
        that exist in the corpus are marked node_type='internal'; unresolved
        external references are node_type='external'. See build_cocitation_network().

    Both graphs can coexist on the same BibNetwork instance.

    Usage
    -----
    net = BibNetwork(bib_dataset)
    net.build_coupling_network()
    net.build_cocitation_network()
    G_coupling  = net.coupling_graph
    G_cocit     = net.cocitation_graph
    """

    def __init__(self, bib_dataset):
        self._dataset         = bib_dataset
        self._coupling_graph  = None
        self._cocitation_graph = None

    # ------------------------------------------------------------------ #
    # Bibliographic coupling                                               #
    # ------------------------------------------------------------------ #

    def build_coupling_network(
        self,
        use_resolved:   bool = True,
        use_unresolved: bool = True,
        min_shared:     int  = 1,
    ) -> 'BibNetwork':
        """Build (or rebuild) the bibliographic coupling graph.

        Parameters
        ----------
        use_resolved : bool
            Use resolved internal reference IDs (requires resolve_references()
            to have been called on the source dataset).
        use_unresolved : bool
            Use raw unresolved reference strings.
        min_shared : int
            Minimum shared references to add an edge between two documents.
        """
        self._coupling_graph = build_coupling_graph(
            self._dataset.dataset,
            use_resolved=use_resolved,
            use_unresolved=use_unresolved,
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

    def build_cocitation_network(
        self,
        use_resolved:    bool = True,
        use_unresolved:  bool = True,
        min_cocitations: int  = 1,
    ) -> 'BibNetwork':
        """Build (or rebuild) the co-citation graph.

        Parameters
        ----------
        use_resolved : bool
            Include resolved internal reference IDs as co-citation nodes.
        use_unresolved : bool
            Include unresolved raw reference strings as co-citation nodes.
        min_cocitations : int
            Minimum co-citation count to add an edge between two references.
        """
        self._cocitation_graph = build_cocitation_graph(
            self._dataset.dataset,
            use_resolved=use_resolved,
            use_unresolved=use_unresolved,
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
    def from_config

    # ------------------------------------------------------------------ #
    # Generic stats                                                        #
    # ------------------------------------------------------------------ #

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
