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

.. autoclass:: pysyrev.core.config.SeedExpansionConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.OpenAlexSeedConfig
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

.. autoclass:: pysyrev.core.config.ReferenceKeysConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.AbstractCompletionConfig
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

topic_report section
--------------------

.. autoclass:: pysyrev.core.config.TopicReportConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReportMetaConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReportSectionsConfig
   :members:
   :show-inheritance:

Report sections
~~~~~~~~~~~~~~~

.. autoclass:: pysyrev.core.config.TopicsSectionConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.BibNetworkSectionConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CouplingPanelConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CocitationPanelConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ReferenceMetadataConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ConnectivityConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.TemporalSectionConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.TopicCharacteristicsConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.TopicSimilarityConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.PaperSelectionConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.CompositeConfig
   :members:
   :show-inheritance:

llm section
-----------

.. autoclass:: pysyrev.core.config.TopicLabelerConfig
   :members:
   :show-inheritance:

download config
---------------

Used by the standalone ``pysyrev download`` sub-command, not by the pipeline.

.. autoclass:: pysyrev.core.config.DownloadConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.UnpaywallConfig
   :members:
   :show-inheritance:

.. autoclass:: pysyrev.core.config.ElsevierDownloadConfig
   :members:
   :show-inheritance:
