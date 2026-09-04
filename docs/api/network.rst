Bibliographic networks
======================

Bibliographic coupling and co-citation networks are no longer a pipeline stage:
they are recomputed from the reviewed corpus' references while the report is
generated, and rendered as panels of the ``topic_report`` section. The functions
below are the building blocks, usable on their own from a notebook.

Both network types share the same machinery — a similarity matrix, Leiden
communities, a force layout and a backbone of the strongest edges — and differ
only in what the matrix counts: shared *references* for coupling, shared
*citers* for co-citation.

Result bundle
-------------

.. autoclass:: pysyrev.core.networks.NetworkResult
   :members:
   :show-inheritance:

Bibliographic coupling
----------------------

.. autofunction:: pysyrev.core.networks.build_coupling
.. autofunction:: pysyrev.core.networks.salton_coupling

Co-citation
-----------

.. autofunction:: pysyrev.core.networks.build_cocitation
.. autofunction:: pysyrev.core.networks.cocitation_matrix
.. autofunction:: pysyrev.core.networks.frequent_references
.. autofunction:: pysyrev.core.networks.reference_texts
.. autofunction:: pysyrev.core.networks.cluster_profiles

Shared machinery
----------------

.. autofunction:: pysyrev.core.networks.reference_sets
.. autofunction:: pysyrev.core.networks.leiden_communities
.. autofunction:: pysyrev.core.networks.force_layout
.. autofunction:: pysyrev.core.networks.backbone_edges
.. autofunction:: pysyrev.core.networks.cluster_terms
.. autofunction:: pysyrev.core.networks.community_topic_crosstab

Inter-community connectivity
----------------------------

.. autofunction:: pysyrev.core.networks.inter_cluster_matrix
.. autofunction:: pysyrev.core.networks.corpus_baseline
.. autofunction:: pysyrev.core.networks.coupling_inout
.. autofunction:: pysyrev.core.networks.groups_from_labels

Figures
-------

.. autofunction:: pysyrev.core.networks.plot_network
.. autofunction:: pysyrev.core.networks.plot_connectivity_matrix
.. autofunction:: pysyrev.core.networks.plot_crosstab_heatmap
