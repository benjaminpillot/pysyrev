Configuration reference
========================

All pipeline behaviour is driven by a single YAML file. Copy
``pysyrev/templates/config.yaml`` and fill in only the sections you need —
absent sections are skipped entirely.

.. code-block:: bash

   # Load programmatically
   from pysyrev.core.config import Config
   cfg = Config.load("my_config.yaml")

Environment variables
---------------------

Any ``${VAR}`` reference in the YAML is resolved against the ``.env`` file
pointed to by the root-level ``env:`` key:

.. code-block:: yaml

   env: /path/to/.env

   bib:
     wos:
       api:
         api_key: ${WOS_API_KEY}


``bib`` — bibliography collection
-----------------------------------

.. code-block:: yaml

   bib:

     wos:
       source: file          # 'file' | 'api'
       file: /path/to/savedrecs.bib   # file path or directory of .bib files
       # api:
       #   api_key: ${WOS_API_KEY}
       #   query: 'ALL=(your query) AND PY=2015-2024'
       #   cache_dir: /path/to/cache/

     open_alex:
       source: api
       api:
         api_key: ${OPENALEX_API_KEY}
         email: ${OPENALEX_EMAIL}
         query: your search terms
         filters:
           publication_year: '2015-2024'
           type: article
         cache_dir: /path/to/cache/

     # scopus: /path/to/scopus.csv
     # pubmed: /path/to/pubmed.nbib

     clean:
       min_signals_to_reject: 2   # garbage signals needed to drop an abstract
       use_langdetect: false

     extract:
       year: 2000                 # min publication year (inclusive)
       language: english          # str, list of str, or blank (keep all)
       # include_doc_type: [article, review]
       # exclude_doc_type: [peer review]

     merge:
       title_similarity: 98       # fuzzy threshold (0–100)

     resolve_references:
       enabled: false             # opt-in — expensive
       flag_unresolved: false

     export:
       export_dir: /path/to/bib_results/
       run_name:                  # blank → auto-timestamp

.. note::

   When ``wos.source: file`` and ``file`` points to a **directory**, all
   ``.bib`` files in that directory are read and concatenated automatically.
   This handles WoS exports split into chunks of 500 or 1 000 records.


``review`` — LLM-based screening
----------------------------------

.. code-block:: yaml

   review:
     doc_dataset:               # blank → auto-detect latest bib run
     export:
       export_dir: /path/to/review_results/
     text_inputs: [title, abstract, keywords]
     inclusion_criteria: |
       <Describe what MUST be true for inclusion.>
     exclusion_criteria: |
       1. <First exclusion reason.>
     decision_rule: majority    # majority | mean
     max_concurrent_requests: 5
     items_per_call: 5
     workflow:
       - round: A
         reviewers: [Reviewer1, Reviewer2]
     reviewers:
       - name: Reviewer1
         provider: anthropic
         model_id: claude-haiku-4-5
         max_tokens: 200
         temperature: 0.7
         reasoning: brief       # brief | cot
         backstory: |
           <Reviewer expertise and role.>


``bib_network`` — bibliographic networks
-----------------------------------------

.. code-block:: yaml

   bib_network:
     doc_dataset:               # blank → auto-detect latest review run
     coupling_network:
       use_resolved: true
       use_unresolved: true
       min_shared: 1
     cocitation_network:
       use_resolved: true
       use_unresolved: true
       min_cocitations: 1
     export:
       export_dir: /path/to/bib_network_results/


``topic_model`` — BERTopic clustering
---------------------------------------

.. code-block:: yaml

   topic_model:
     doc_dataset:               # blank → auto-detect latest review run
     export:
       export_dir: /path/to/topic_modeling/
     distance: euclidean        # euclidean | chebyshev
     keep_n_results: 10
     hdbscan:
       min_topic_size_range: [10, 50]
       min_sample_range: [2, 10]
       topic_size_step: 4
       min_sample_step: 2
     umap:
       n_neighbors: [5, 10]
       n_components: [5, 10, 15]


``topic_report`` — PDF report generation
-----------------------------------------

.. code-block:: yaml

   topic_report:
     model_index: 0             # index in best_results.csv (0 = best)
     export_to: /path/to/report/
     # run_dir:                 # blank → auto-detect latest topic_model run


``llm`` — topic label generation
----------------------------------

When present, an LLM generates human-readable labels for each topic:

.. code-block:: yaml

   llm:
     provider: anthropic
     model_id: claude-haiku-4-5-20251001
     max_tokens: 200
     temperature: 0.3
     n_repr_docs_for_labeling: 3


``report`` — PDF layout
------------------------

.. code-block:: yaml

   report:
     meta:
       title: My Literature Review
       author: My Name
       date_format: "%d/%m/%Y"
     sections:
       topics:
         n_repr_docs_per_topic: 5
       temporal:
         variants: [absolute, cumulative, normalized, weighted]
       topic_characteristics:
         n_top_cited_per_topic: 5
         n_top_cited_global: 50
       paper_selection:
         min_year: 2020
         proportion_per_topic: 0.15
         selection_by: citations   # citations | random
         export_annex: true
         annex_format: csv         # csv | txt
