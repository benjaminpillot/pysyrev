pysyrev — Python Systematic Review
====================================

**pysyrev** is an automated, LLM-assisted PRISMA workflow for systematic
literature reviews. It covers the full pipeline — from raw bibliographic
records to screened, deduplicated, and thematically structured corpora —
and produces a PDF report at the end.

Features
--------

- **Multi-source ingestion** — Web of Science (file or REST API), OpenAlex
  (file, REST API, or seed-driven expansion), Scopus, PubMed
- **Automatic deduplication** — fuzzy title matching across sources, with a
  configurable source priority; duplicates are aliased so references still
  resolve
- **Canonical reference keys** — optionally remap every reference onto a shared
  DOI key space so bibliographic coupling holds across merged sources
  regardless of merge order
- **LLM-based screening** — multi-reviewer workflows with majority or mean
  voting, powered by any provider supported by LiteLLM (Anthropic, OpenAI,
  Ollama, LiteLLM proxy…) or by `Albert <https://albert.api.etalab.gouv.fr>`_,
  the French State's sovereign gateway
- **Deferred batch reviewing** — screen the corpus through the provider's batch
  endpoint (``review.use_batch_api``): same prompts, results within 24 h, half
  price; interrupted runs re-attach to the batch already paid for
- **Token cost estimation** — price the review stage before running it
  (``pysyrev estimate``)
- **Topic modelling** — BERTopic-based clustering with UMAP + HDBSCAN grid
  search, ranked by coherence scores
- **Bibliographic network panels** — coupling and co-citation networks (Salton
  similarity + Leiden communities, readable backbone layout) cross-coloured by
  BERTopic topic, plus an inter-topic connectivity matrix
- **Reading-list selection** — rank the most relevant papers per topic by
  citations, network centrality, or a three-axis composite
- **PDF report generation** — declarative, theme-aware PDF engine built on
  ReportLab
- **Full-text download** — retrieve PDFs for a candidate list
  (``pysyrev download``), cascading Unpaywall → OpenAlex → Elsevier TDM

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
   * - Topic modelling
     - ``topic-model``
     - Cluster documents into topics using BERTopic; rank configurations by
       coherence
   * - Report
     - ``topic-report``
     - Generate a PDF report from the selected topic model run, including the
       bibliographic coupling, co-citation and topic-connectivity network
       panels (recomputed from the reviewed corpus' references)

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
