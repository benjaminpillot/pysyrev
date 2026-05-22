Configuration classes
======================

These dataclasses mirror the YAML structure. They are produced by
:meth:`Config.load() <pysyrev.core.config.Config.load>` and consumed by
the runtime pipeline classes.

Root
----

.. autoclass:: pysyrev.core.config.Config
   :members:
   :show-inheritance:

bib section
-----------

.. autoclass:: pysyrev.core.config.BibConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.WosSourceConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.WosApiConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.OpenAlexSourceConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.OpenAlexApiConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CleanConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ExtractConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.MergeConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ResolveReferencesConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.BibExportConfig
   :members:
   :show-inheritance:

review section
--------------

.. autoclass:: pysyrev.core.config.ReviewConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReviewerConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReviewExportConfig
   :members:
   :show-inheritance:

bib_network section
--------------------

.. autoclass:: pysyrev.core.config.BibNetworkConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CouplingNetworkConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CocitationNetworkConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.BibNetworkExportConfig
   :members:
   :show-inheritance:

topic_model section
--------------------

.. autoclass:: pysyrev.core.config.TopicModelConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.HDBSCANConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.UMAPConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.BertopicConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.TopicExportConfig
   :members:
   :show-inheritance:

topic_report / llm / report sections
--------------------------------------

.. autoclass:: pysyrev.core.config.TopicReportConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.TopicLabelerConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReportConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReportMetaConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReportSectionsConfig
   :members:
   :show-inheritance:
