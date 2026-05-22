Quick start
===========

CLI
---

.. code-block:: bash

   # Run all stages declared in the config
   python -m pysyrev config.yaml

   # Or via the installed entry point
   pysyrev config.yaml

   # Run a single stage
   python -m pysyrev config.yaml --stage bib
   python -m pysyrev config.yaml --stage review
   python -m pysyrev config.yaml --stage bib-network
   python -m pysyrev config.yaml --stage topic-model
   python -m pysyrev config.yaml --stage topic-report

Python API
----------

.. code-block:: python

   from pysyrev import Pipeline

   # Full pipeline in one call
   pipeline = Pipeline.from_config("config.yaml")
   pipeline.run()

   # Or stage by stage — results persist between calls
   pipeline.run(stages=["bib"])
   pipeline.run(stages=["review"])       # uses pipeline.bib.dataset automatically
   pipeline.run(stages=["topic-report"]) # generates the PDF report

   # Access results
   df_all  = pipeline.bib.dataset           # pd.DataFrame — all collected documents
   df_kept = pipeline.review.included_docs  # pd.DataFrame — LLM-screened inclusions
   network = pipeline.network               # BibNetwork
   topic   = pipeline.topic                 # TopicModel
   report  = pipeline.report                # TopicReport

Report-only run
---------------

A config containing only ``topic_report`` (and optionally ``report`` and
``llm``) is valid. This lets you regenerate a report from a previous
topic-model run without re-running the full pipeline:

.. code-block:: yaml
   :caption: report_only.yaml

   topic_report:
     run_dir: /path/to/topic_modeling/run_2026-05-01T120000/
     model_index: 0
     export_to: /path/to/output/report/

.. code-block:: bash

   python -m pysyrev report_only.yaml

Auto-detection between stages
------------------------------

When ``doc_dataset`` or ``run_dir`` fields are left blank, pysyrev
auto-detects the most recent output of the previous stage:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Blank field
     - Auto-detected from
   * - ``review.doc_dataset``
     - latest run in ``bib.export.export_dir``
   * - ``bib_network.doc_dataset``
     - latest run in ``review.export.export_dir``
   * - ``topic_model.doc_dataset``
     - latest run in ``review.export.export_dir``
   * - ``topic_report.run_dir``
     - latest run in ``topic_model.export.export_dir``
