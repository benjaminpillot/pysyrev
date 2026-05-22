pysyrev — Python Systematic Review
====================================

**pysyrev** is an automated, LLM-assisted PRISMA workflow for systematic
literature reviews. It covers the full pipeline — from raw bibliographic
records to screened, deduplicated, and thematically structured corpora —
and produces a PDF report at the end.

Features
--------

- **Multi-source ingestion** — Web of Science (file or REST API), OpenAlex
  (file or REST API), Scopus, PubMed
- **Automatic deduplication** — fuzzy title matching across sources
- **LLM-based screening** — multi-reviewer workflows with majority or mean
  voting, powered by any provider supported by LiteLLM (Anthropic, OpenAI,
  Ollama, LiteLLM proxy…)
- **Bibliographic network analysis** — bibliographic coupling and co-citation
  graphs exported as GraphML
- **Topic modelling** — BERTopic-based clustering with UMAP + HDBSCAN grid
  search, ranked by coherence scores
- **PDF report generation** — declarative, theme-aware PDF engine built on
  ReportLab

Pipeline stages
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Stage
     - Key
     - Description
   * - Bibliography
     - ``bib``
     - Fetch, clean, filter, deduplicate, and optionally resolve references
   * - LLM review
     - ``review``
     - Screen documents against inclusion/exclusion criteria with one or more
       LLM reviewers
   * - Bibliographic network
     - ``bib-network``
     - Build coupling and co-citation networks from the included corpus
   * - Topic modelling
     - ``topic-model``
     - Cluster documents into topics using BERTopic; rank configurations by
       coherence
   * - Report
     - ``topic-report``
     - Generate a PDF report from the selected topic model run

All sections are optional — only the stages declared in the config file are
executed. Each stage auto-detects the most recent output of the previous one
when run standalone.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   configuration
   api/index
