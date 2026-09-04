Quick start
===========

CLI
---

.. code-block:: bash

   # Run all stages declared in the config
   pysyrev config.yaml

   # Or, if the entry point is not on PATH
   python -m pysyrev config.yaml

   # Run one or more specific stages (always executed in canonical order)
   pysyrev config.yaml --stage bib
   pysyrev config.yaml --stage topic-model topic-report

   # Run from a given stage to the end (all configured stages from there on)
   pysyrev config.yaml --from topic-model

Valid stage names: ``bib`` | ``review`` | ``topic-model`` | ``topic-report``.
``--stage`` and ``--from`` are mutually exclusive.

Sub-commands
------------

.. code-block:: bash

   # Price the review stage before paying for it (counts tokens, calls no model)
   pysyrev estimate config.yaml
   pysyrev estimate config.yaml --sweep      # what other items_per_call would cost
   pysyrev estimate config.yaml --calibrate previous_run/reviewed_total.csv
   pysyrev estimate config.yaml --offline    # no API call; ±15 % from character counts

   # Download full-text papers for a candidate list
   # (cascade: Unpaywall → OpenAlex → Elsevier TDM)
   pysyrev download candidates.csv output_folder/
   pysyrev download candidates.csv output_folder/ --config download_config.yaml

Python API
----------

.. code-block:: python

   from pysyrev import Pipeline, ALL_STAGES

   # Full pipeline in one call — runs only the stages declared in the config
   pipeline = Pipeline.from_config("config.yaml")
   pipeline.run()

   # Or stage by stage — results persist between calls
   pipeline.run(stages=["bib"])
   pipeline.run(stages=["review"])       # uses pipeline.bib.dataset automatically
   pipeline.run(stages=["topic-report"]) # generates the PDF report

   # Run from a given stage to the end
   start = ALL_STAGES.index("topic-model")
   pipeline.run(stages=ALL_STAGES[start:])

   # Access results
   df_all  = pipeline.bib.dataset           # pd.DataFrame — all collected documents
   df_kept = pipeline.review.included_docs  # pd.DataFrame — LLM-screened inclusions
   topic   = pipeline.topic                 # TopicModel
   report  = pipeline.report                # TopicReport

Report-only run
---------------

A config containing only ``topic_report`` (and optionally ``llm``) is valid.
This lets you regenerate a report from a previous topic-model run without
re-running the full pipeline:

.. code-block:: yaml
   :caption: report_only.yaml

   topic_report:
     run_dir: /path/to/topic_modeling/run_2026-05-01T120000/   # or blank to auto-detect
     export_to: /path/to/output/report/

.. code-block:: bash

   pysyrev report_only.yaml

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
   * - ``topic_model.doc_dataset``
     - latest run in ``review.export.export_dir`` (also the source the
       report's network panels are recomputed from)
   * - ``topic_report.run_dir``
     - latest run in ``topic_model.export.export_dir``
