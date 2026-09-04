Configuration reference
========================

All pipeline behaviour is driven by a single YAML file. Copy
``pysyrev/templates/config.yaml`` and fill in only the sections you need —
absent sections are skipped entirely.

.. code-block:: bash

   # Load and run programmatically
   from pysyrev.core.config import Config
   from pysyrev import Pipeline

   cfg = Config.load("my_config.yaml")
   Pipeline.from_config("my_config.yaml").run()

.. rubric:: Stage execution order

Sections are executed in canonical order, regardless of their order in the
YAML file:

.. code-block:: text

   bib  →  review  →  topic_model  →  topic_report


.. rubric:: Output auto-wiring

When ``doc_dataset`` / ``run_dir`` fields are left blank, ``Config.load()``
automatically propagates outputs between stages:

- ``bib.export.export_dir``         → ``review.doc_dataset``
- ``review.export.export_dir``      → ``topic_model.doc_dataset``
- ``topic_model.export.export_dir`` → ``topic_report.run_dir``

.. note::
   ``topic_model.doc_dataset`` is also the file the report's bibliographic
   network panels are recomputed from, so it must carry the corpus' reference
   columns even when you only want a report.


----


Root level
----------

``env``
   :Type: ``str``
   :Default: —
   :Required: no

   Path to a ``.env`` file.  Any ``${VAR}`` reference found anywhere in the
   YAML is resolved against this file at load time.  Variables already set in
   the process environment take precedence.

   .. code-block:: yaml

      env: /path/to/.env

      bib:
        wos:
          api:
            api_key: ${WOS_API_KEY}


----


``bib`` — bibliography collection
-----------------------------------

The ``bib`` section collects records from one or more bibliographic sources,
cleans and filters them, removes cross-source duplicates, and writes a
consolidated CSV.

``wos``
   :Type: mapping or ``str``
   :Default: —
   :Required: no

   Web of Science source. Can be a plain file path (shorthand for
   ``source: file``) or a structured block with the keys below.

   .. note::
      When ``source: file`` and ``file`` points to a **directory**, all
      ``.bib`` files in that directory are concatenated automatically.
      This handles WoS exports split into chunks of 500 or 1 000 records.

   ``wos.source``
      :Type: ``str``
      :Default: ``file``
      :Values: ``file`` | ``api``
      :Required: yes

      Whether to read from a local export file or from the WoS Expanded API.

   ``wos.file``
      :Type: ``str``
      :Default: —
      :Required: when ``source: file``

      Path to a ``.bib`` export file, or to a directory containing multiple
      ``.bib`` files (chunked WoS export).

   ``wos.api.api_key``
      :Type: ``str``
      :Default: —
      :Required: when ``source: api``

      WoS Expanded API key.  Use ``${WOS_API_KEY}`` to read from the
      ``.env`` file.

   ``wos.api.query``
      :Type: ``str``
      :Default: —
      :Required: when ``source: api``

      WoS Query Language expression, e.g.
      ``'ALL=(agent-based model) AND PY=2015-2024'``.

   ``wos.api.cache_dir``
      :Type: ``str``
      :Default: ``null`` (no caching)
      :Required: no

      Local directory where raw API responses are cached.  Subsequent runs
      with the same query read from disk instead of hitting the API.

``open_alex``
   :Type: mapping or ``str``
   :Default: —
   :Required: no

   OpenAlex source.  Can be a plain CSV file path or a structured block.

   ``open_alex.source``
      :Type: ``str``
      :Default: ``file``
      :Values: ``file`` | ``api`` | ``seed``
      :Required: yes

   ``open_alex.file``
      :Type: ``str``
      :Default: —
      :Required: when ``source: file``

      Path to an OpenAlex CSV export.

   ``open_alex.api.api_key``
      :Type: ``str``
      :Default: —
      :Required: when ``source: api``

   ``open_alex.api.email``
      :Type: ``str``
      :Default: ``null``
      :Required: no

      Providing an e-mail address enables the OpenAlex *polite pool*
      (higher rate limits).  Strongly recommended for non-trivial usage.

   ``open_alex.api.query``
      :Type: ``str``
      :Default: ``null``
      :Required: no (one of ``query`` or ``filters`` must be set)

      Free-text BM25 search on title and abstract.

   ``open_alex.api.filters``
      :Type: mapping
      :Default: ``null``
      :Required: no

      Structured OpenAlex filters, combined with ``AND``.  Common keys:

      .. code-block:: yaml

         filters:
           publication_year: '2015-2024'
           type: article

   ``open_alex.api.cache_dir``
      :Type: ``str``
      :Default: ``null`` (no caching)
      :Required: no

   ``open_alex.seed``
      :Type: mapping
      :Default: —
      :Required: when ``source: seed``

      **Seed expansion** — build the corpus from a handful of *seed papers*
      instead of a query.  The candidate pool is the union of four arms: the
      seeds themselves, the papers **citing** them, the papers they **cite**,
      and optional text queries.  Use it when a subfield has no reliable
      keyword handle, or to complement a query-based source (declare both and
      let ``merge`` deduplicate).

      The pool is deliberately over-collected: it is a set of *candidates*, not
      a corpus.  ``extract`` filters it on year, document type and language,
      and the ``review`` stage screens it for scope.

      .. code-block:: yaml

         open_alex:
           source: seed
           seed:
             api_key: ${OPENALEX_API_KEY}
             email: ${OPENALEX_EMAIL}
             file: seeds.txt                 # one DOI per line
             queries:
               - agent-based energy transition
               - energy justice modelling
             year_min: 2000

   ``open_alex.seed.file``
      :Type: ``str``
      :Default: ``null``
      :Required: one of ``file`` / ``seeds``

      Text file listing the seeds, **one DOI per line**.  An OpenAlex work id
      (``W2741809807``) or a ``doi.org`` / ``openalex.org`` URL works too.
      Blank lines and lines starting with ``#`` are ignored; anything after the
      identifier on a line (a title, a note) is skipped.

   ``open_alex.seed.seeds``
      :Type: list of ``str``
      :Default: ``null``
      :Required: one of ``file`` / ``seeds``

      Inline seed list, merged with the identifiers read from ``file``.

   ``open_alex.seed.queries``
      :Type: list of ``str``
      :Default: ``null``
      :Renamed: was ``phrases`` — the old spelling is still accepted

      Title/abstract searches in the field's own vocabulary (10–25 is a good
      range).  They are the recall backstop: they reach papers that neither
      cite nor are cited by a seed.  Omit for a citation-only expansion.

      Each entry is *one* OpenAlex search, run with the same engine and the
      same syntax as an ``api.query``: its terms are **AND-ed**, not matched
      as a phrase, and matching is stemmed and insensitive to hyphens and
      plurals (``agent based energy transitions`` and ``agent-based energy
      transition`` return the same set).  Quote the entry for a phrase match
      — the words must then appear in that order, though still stemmed, so no
      match is ever literal — which is far narrower::

          - agent-based energy transition        # ~4800 works
          - '"agent-based energy transition"'    # 2 works

      ``AND`` / ``OR`` / ``NOT`` are supported inside an entry.  Results are
      relevance-ranked, so ``max_per_query`` keeps the *best* matches, not
      an arbitrary slice.  The entries' result sets are then unioned into the
      pool.

      Three characters are reserved by the OpenAlex filter syntax and do not
      mean what they appear to: a comma is stripped (it would otherwise be
      read as a filter separator), ``|`` is an OR between filter values, and
      a leading ``!`` negates the search.  Spell a disjunction ``OR`` rather
      than ``|``.

   ``open_alex.seed.year_min`` / ``open_alex.seed.year_max``
      :Type: ``int``
      :Default: ``null`` (no bound)

      Publication-year window used as a *retrieval* bound on the arms that
      support it.  Definitive filtering stays with ``extract``, so seeds
      outside the window still drive the expansion.

   ``open_alex.seed.use_forward`` / ``open_alex.seed.use_backward``
      :Type: ``bool``
      :Default: ``true``

      Toggle the citation arms (papers citing the seeds / papers the seeds
      cite).  Both off leaves a query-only pool — a plain keyword corpus,
      useful as a comparison baseline.

   ``open_alex.seed.max_per_seed`` / ``open_alex.seed.max_per_query``
      :Type: ``int``
      :Default: ``600`` / ``400``
      :Renamed: ``max_per_phrase`` — the old spelling is still accepted

      Caps on the records pulled per seed (forward citations) and per text
      query.  Raise for exhaustive recall, lower for a quick draft: a
      highly-cited seed can otherwise return tens of thousands of papers.

   ``open_alex.seed.api_key`` / ``open_alex.seed.email`` / ``open_alex.seed.cache_dir``
      Same meaning as their ``open_alex.api`` counterparts.  These credentials
      are also reused by ``reference_keys`` when the source is seed-based.

``scopus``
   :Type: ``str``
   :Default: —
   :Required: no

   Path to a Scopus CSV export.  File-only: there is no Scopus API source.

``pubmed``
   :Type: ``str``
   :Default: —
   :Required: no

   Path to a PubMed export (MEDLINE ``.nbib`` / ``.txt``).  File-only.

``clean``
   :Type: mapping
   :Default: all defaults applied
   :Required: no

   Abstract quality filter applied before document extraction.

   ``clean.min_signals_to_reject``
      :Type: ``int``
      :Default: ``2``

      Number of garbage signals (boilerplate patterns, encoding artefacts,
      etc.) that must be detected before an abstract is dropped.  Raising
      this value makes the filter more permissive.

   ``clean.extra_garbage_phrases``
      :Type: list of ``str``
      :Default: ``[]``

      Additional literal phrases that count as garbage signals.

   ``clean.max_abstract_chars``
      :Type: ``int``
      :Default: ``null`` (no truncation)

      Cut abstracts longer than this, at a sentence boundary when possible.
      Some sources return the whole full text in the abstract field; left
      whole, one such record can dominate a review call's token budget.

   ``clean.use_langdetect``
      :Type: ``bool``
      :Default: ``false``

      When ``true``, records whose abstract language cannot be confirmed
      by ``langdetect`` are flagged.  Disable when processing multilingual
      corpora or when abstracts are absent.

``extract``
   :Type: mapping
   :Default: all defaults applied
   :Required: no

   Document-level filtering applied after cleaning.

   ``extract.year``
      :Type: ``int``
      :Default: ``1900``

      Minimum publication year (inclusive).  Records published before this
      year are dropped.

   ``extract.language``
      :Type: ``str`` or list of ``str``
      :Default: ``null`` (keep all languages)

      Language or list of languages to keep, e.g. ``english`` or
      ``[english, french]``.

   ``extract.nb_citations``
      :Type: ``int``
      :Default: ``0``

      Minimum citation count (inclusive).  Records with fewer citations
      are dropped.

   ``extract.include_doc_type``
      :Type: list of ``str``
      :Default: ``null`` (keep all types)

      Whitelist of document types to retain, e.g.
      ``[article, review]``.  Takes lower priority than
      ``exclude_doc_type``.

   ``extract.exclude_doc_type``
      :Type: list of ``str``
      :Default: ``null``

      Document types to remove (fuzzy-matched).  Takes priority over
      ``include_doc_type``.  Example: ``[peer review, retraction]``.

   ``extract.scorer``
      :Type: ``str``
      :Default: ``partial_token_sort_ratio``
      :Values: any ``rapidfuzz`` scorer name

      Fuzzy scorer used for document-type matching.

   ``extract.score_cutoff``
      :Type: ``int``
      :Default: ``90``
      :Range: 0–100

      Minimum fuzzy score for a document type to match.

``merge``
   :Type: mapping
   :Default: all defaults applied
   :Required: no

   Cross-source duplicate removal (based on title similarity).

   ``merge.title_similarity``
      :Type: ``int``
      :Default: ``98``
      :Range: 0–100

      Fuzzy-match threshold for two titles to be considered duplicates.
      Lower values increase recall but risk false positives.

   ``merge.ngram_size``
      :Type: ``int``
      :Default: ``3``

      Character n-gram size used to build the candidate index.

   ``merge.max_candidates_per_row``
      :Type: ``int``
      :Default: ``200``

      Maximum number of candidate duplicates inspected per record.
      Increase for large corpora if recall is insufficient.

   ``merge.priority``
      :Type: list of ``str``
      :Default: ``[open_alex, wos, scopus, pubmed]``

      Which source wins when the same document is found twice.  The default
      keeps OpenAlex records for their stable ids; the losing duplicates are
      aliased, so references pointing at them still resolve.

   ``merge.scorer``
      :Type: ``str``
      :Default: ``token_set_ratio``
      :Values: any ``rapidfuzz`` scorer name

``resolve_references``
   :Type: mapping
   :Default: ``enabled: false``
   :Required: no

   Cross-record reference resolution (opt-in, expensive).  Links each
   cited reference string to a known record in the corpus.

   ``resolve_references.enabled``
      :Type: ``bool``
      :Default: ``false``

      Set to ``true`` to activate reference resolution.  This step is
      computationally intensive on large corpora.

   ``resolve_references.flag_unresolved``
      :Type: ``bool``
      :Default: ``false``

      When ``true``, references that cannot be matched are annotated in
      the output rather than silently dropped.

   ``resolve_references.fuzzy_score_cutoff``
      :Type: ``int``
      :Default: ``90``
      :Range: 0–100

      Minimum fuzzy score for a reference string to be accepted as a match.

   ``resolve_references.ngram_size``
      :Type: ``int``
      :Default: ``3``

   ``resolve_references.max_candidates``
      :Type: ``int``
      :Default: ``50``

      Maximum candidate records examined per reference string.

   ``resolve_references.scorer``
      :Type: ``str``
      :Default: ``token_set_ratio``
      :Values: any ``rapidfuzz`` scorer name

``abstract_completion``
   :Type: list of mappings
   :Default: ``null``
   :Required: no

   Fill abstracts the primary source left empty, before ``clean`` drops those
   records.  Entries run in order, so a later provider only sees what the
   earlier ones could not fill.  Nothing but the ``abstract`` column is
   touched — the ids and reference lists the coupling network needs are
   preserved.

   .. code-block:: yaml

      abstract_completion:
        - provider: wos
          api_key: ${WOS_API_KEY}
          cache_dir: /path/to/wos_cache

   ``abstract_completion[].provider``
      :Type: ``str``
      :Required: yes
      :Values: ``wos``

   ``abstract_completion[].api_key``
      :Type: ``str``
      :Default: ``null``

   ``abstract_completion[].cache_dir``
      :Type: ``str``
      :Default: ``null``

``reference_keys``
   :Type: mapping
   :Default: ``null`` (references keep their per-source form)
   :Required: no

   Remap every reference onto a shared DOI key space, written to a
   ``reference_keys`` column.  Without it, bibliographic coupling across merged
   sources depends on the merge order: a WoS record and an OpenAlex record can
   cite the same paper under two different reference strings and never couple.

   Intra-corpus references and references that already carry a DOI map for
   free; extra-corpus OpenAlex ids are resolved to DOIs through the OpenAlex
   API and cached.  The resolution step is skipped automatically when there is
   no other-source DOI to bridge to.

   ``reference_keys.cache``
      :Type: ``str``
      :Default: ``null``

      JSON cache of resolved ``openalex id → DOI`` pairs, reused across runs.

   ``reference_keys.resolve_external``
      :Type: ``bool``
      :Default: ``true``

      Call the OpenAlex API for references outside the corpus.  Set to
      ``false`` to stay offline and key only what can be mapped locally.

``export`` *(bib)*
   :Type: mapping
   :Required: yes

   ``export.export_dir``
      :Type: ``str``
      :Required: yes

      Parent directory for bib stage outputs.  Each run is stored in a
      sub-directory ``<export_dir>/<run_name>/bib_dataset.csv``.

   ``export.run_name``
      :Type: ``str``
      :Default: ``null`` → auto-generated timestamp ``YYYY-MM-DDTHHMMSS``

      A human-readable label for the run, e.g. ``may_2026_wos_oa``.
      Re-using an existing name reopens that run directory.

   ``export.dataset``
      :Type: ``str``
      :Default: ``bib_dataset.csv``

      File name of the consolidated corpus written inside the run directory.


----


``review`` — LLM-based title/abstract screening
-------------------------------------------------

Runs a multi-reviewer LLM workflow to decide whether each record should be
included in the review.

``review.doc_dataset``
   :Type: ``str``
   :Default: ``null`` (auto-detect latest bib run)

   Path to a ``bib_dataset.csv`` produced by the ``bib`` stage.  Leave
   blank to pick up the most recent file in
   ``bib.export.export_dir`` automatically.

``review.text_inputs``
   :Type: list of ``str``
   :Default: —
   :Required: yes
   :Values: any subset of ``[title, abstract, keywords]``

   Fields sent to the LLM for each record.

``review.inclusion_criteria``
   :Type: ``str`` (multi-line)
   :Default: —
   :Required: yes

   Free-text description of what **must** be true for a document to be
   included.  Passed verbatim to every reviewer.

``review.exclusion_criteria``
   :Type: ``str`` (multi-line)
   :Default: —
   :Required: yes

   Free-text list of reasons to **exclude** a document.  Passed verbatim
   to every reviewer.

``review.decision_rule``
   :Type: ``str``
   :Default: ``majority``
   :Values: ``majority`` | ``mean``

   How individual reviewer verdicts are aggregated into a final decision.
   ``majority`` requires more than half of reviewers to agree;
   ``mean`` averages their numerical scores.

``review.batch_size``
   :Type: ``int``
   :Default: ``100``

   Number of records processed between checkpoint saves.  Smaller values
   reduce data loss on interruption; larger values reduce overhead.

``review.api_pause``
   :Type: ``float``
   :Default: ``30.0``

   Pause in seconds between batches.  Acts as a rate-limit guard for
   hosted APIs.

``review.sample_size``
   :Type: ``int``
   :Default: ``null`` (process full dataset)

   If set, a random sample of this size is drawn from the dataset.
   Useful for pilot runs.

``review.sample_seed``
   :Type: ``int``
   :Default: ``null`` (a fresh sample every run)

   Seed for the ``sample_size`` draw.  Set it to make a test run reproducible.

``review.max_retries``
   :Type: ``int``
   :Default: ``null`` → module default (2)

   Section-level default for API call retries on error.  Can be
   overridden per reviewer.

``review.max_concurrent_requests``
   :Type: ``int``
   :Default: ``null`` → module default (10)

   Section-level default for concurrent API requests.  Keep 5–10 for
   Anthropic, up to 30 for OpenAI.  Can be overridden per reviewer.

``review.items_per_call``
   :Type: ``int``
   :Default: ``null`` → module default (1)

   Number of records sent per API call.  Batching records reduces cost;
   the backstory is sent only once per call.  Can be overridden per
   reviewer.

``review.requests_per_minute``
   :Type: ``int``
   :Default: ``null`` → provider default (Albert: 10; others: unpaced)

   Client-side quota, shared by every reviewer and by the ``llm`` labeller
   running on the same key — a gateway counts requests per key, not per
   reviewer, so pacing each reviewer separately overshoots as soon as there is
   more than one.  A slot is taken at the moment a request is sent, and a 429
   makes the whole run wait rather than burning a retry.

``review.timeout``
   :Type: ``float``
   :Default: ``null`` → provider default

   Seconds allowed per call.  Overridable per reviewer.

``review.use_batch_api``
   :Type: ``bool``
   :Default: ``false``

   Screen the corpus through the provider's batch endpoint instead of live
   calls: same prompts, results within 24 h, half price on input and output.
   The run blocks until the batch completes, and an interrupted run re-attaches
   to the batch already paid for rather than resubmitting it.  Supported on
   Anthropic; other providers fall back to live calls.

``review.batch_poll_interval``
   :Type: ``float``
   :Default: ``30.0``

   Seconds between batch status checks when ``use_batch_api`` is on.

``review.export`` *(review)*
   :Type: mapping
   :Required: yes

   ``export.export_dir``
      :Type: ``str``
      :Required: yes

      Parent directory for review outputs.  Each run produces
      ``reviewed_included.csv`` and ``reviewed_total.csv``.

   ``export.run_name``
      :Type: ``str``
      :Default: ``null`` → auto-generated timestamp

   ``export.cache_dir``
      :Type: ``str``
      :Default: ``null`` → ``<run_dir>/cache/``

      Directory for LLM response caching between runs.

``review.workflow``
   :Type: list of round mappings
   :Required: yes

   Ordered list of screening rounds.  Each round specifies a label and
   the reviewers that participate.  Round N+1 only processes records where
   round N produced no consensus.

   .. code-block:: yaml

      workflow:
        - round: A
          reviewers: [Reviewer1, Reviewer2]
        - round: B          # optional tie-breaker
          reviewers: [Reviewer3]

   ``round``
      :Type: ``str``

      Arbitrary label for the round (e.g. ``A``, ``B``, ``pilot``).

   ``reviewers``
      :Type: list of ``str``

      Names of reviewers participating in this round.  Must match names
      declared in ``review.reviewers``.

``review.reviewers``
   :Type: list of reviewer mappings
   :Required: yes

   Each entry defines one LLM reviewer.

   ``name``
      :Type: ``str``
      :Required: yes

      Unique identifier for this reviewer.  Referenced in ``workflow``.

   ``provider``
      :Type: ``str``
      :Required: yes
      :Values: ``anthropic`` | ``openai`` | ``albert`` | ``litellm`` | ``ollama``

      LLM provider.  Use ``litellm`` or ``ollama`` for custom or
      self-hosted endpoints.

   ``model_id``
      :Type: ``str``
      :Required: yes

      Model identifier as accepted by the provider, e.g.
      ``claude-haiku-4-5`` or ``gpt-4o-mini``.

   ``max_tokens``
      :Type: ``int``
      :Required: yes

      Maximum tokens in the model's response.  200 is usually sufficient
      for a verdict + brief justification.

   ``temperature``
      :Type: ``float``
      :Required: yes
      :Range: 0.0–2.0

      Sampling temperature.  Lower values produce more deterministic
      verdicts; ``0.1`` is appropriate for conservative reviewers.

   ``backstory``
      :Type: ``str`` (multi-line)
      :Required: yes

      Reviewer persona: domain expertise, role, reviewing style.
      Injected as the system prompt.

   ``reasoning``
      :Type: ``str``
      :Default: ``brief``
      :Values: ``brief`` | ``cot``

      ``brief`` asks for a short justification; ``cot`` requests a
      full chain-of-thought before the verdict.

   ``host``
      :Type: ``str``
      :Default: ``null`` (use default hosted endpoint)

      Custom API endpoint.  Required for ``litellm`` and ``ollama``;
      leave blank for Anthropic / OpenAI.

   ``reasoning_effort``
      :Type: ``str``
      :Default: ``null``
      :Values: ``low`` | ``medium`` | ``high``

      Extended-thinking effort level.  Only applicable to models that
      support extended thinking (e.g. ``claude-sonnet-4-5``).

   ``additional_context``
      :Type: ``str``
      :Default: ``null``

      Extra context appended to each prompt, e.g. the verdicts of
      previous reviewers for a tie-breaker round.

   ``max_retries``
      :Type: ``int``
      :Default: ``null`` → section-level ``review.max_retries``

   ``max_concurrent_requests``
      :Type: ``int``
      :Default: ``null`` → section-level ``review.max_concurrent_requests``

   ``items_per_call``
      :Type: ``int``
      :Default: ``null`` → section-level ``review.items_per_call``

   ``requests_per_minute``
      :Type: ``int``
      :Default: ``null`` → section-level ``review.requests_per_minute``

   ``timeout``
      :Type: ``float``
      :Default: ``null`` → section-level ``review.timeout``


----


``topic_model`` — BERTopic clustering
---------------------------------------

Runs a grid search over HDBSCAN and UMAP hyperparameters using BERTopic,
scores each configuration, and writes the top ``keep_n_results``
configurations to ``best_results.csv``.

``topic_model.doc_dataset``
   :Type: ``str``
   :Default: ``null`` (auto-detect latest review run)

``topic_model.distance``
   :Type: ``str``
   :Default: ``euclidean``
   :Values: ``euclidean`` | ``chebyshev``

   Distance metric used to rank hyperparameter configurations in the
   multi-objective scoring space.

``topic_model.keep_n_results``
   :Type: ``int``
   :Default: ``10``

   Number of best-ranked configurations saved to ``best_results.csv``.

``topic_model.coherence_scorer``
   :Type: mapping
   :Default: all defaults applied

   ``coherence_scorer.ranking``
      :Type: ``str``
      :Default: ``u_mass``
      :Values: any ``gensim`` coherence measure

      Fast coherence metric used to rank all configurations in the grid.

   ``coherence_scorer.purity``
      :Type: ``str``
      :Default: ``c_v``
      :Values: any ``gensim`` coherence measure

      Slower, higher-quality metric applied only to the top-ranked
      configurations to compute a purity score.

``topic_model.hdbscan``
   :Type: mapping
   :Default: minimal grid (single point ``[2, 2]``)

   HDBSCAN grid-search parameters.

   ``hdbscan.min_topic_size_range``
      :Type: list of two ``int``
      :Default: ``[2, 2]``

      ``[min, max]`` bounds for the ``min_cluster_size`` grid.

   ``hdbscan.min_sample_range``
      :Type: list of two ``int``
      :Default: ``[2, 2]``

      ``[min, max]`` bounds for the ``min_samples`` grid.

   ``hdbscan.topic_size_step``
      :Type: ``int``
      :Default: ``1``

      Step size for the ``min_cluster_size`` axis of the grid.

   ``hdbscan.min_sample_step``
      :Type: ``int``
      :Default: ``1``

      Step size for the ``min_samples`` axis of the grid.

   ``hdbscan.cluster_selection_method``
      :Type: ``str``
      :Default: ``leaf``
      :Values: ``eom`` | ``leaf``

   ``hdbscan.metric``
      :Type: ``str``
      :Default: ``euclidean``

      Distance metric passed to HDBSCAN.

   ``hdbscan.prediction_data``
      :Type: ``bool``
      :Default: ``true``

      Precompute data structures for soft cluster membership prediction.

``topic_model.umap``
   :Type: mapping
   :Default: minimal grid (single point ``[5]``, ``[5]``)

   UMAP grid-search parameters.  Each field accepts a list of values;
   all combinations are explored.

   ``umap.n_neighbors``
      :Type: list of ``int``
      :Default: ``[5]``

      Candidate values for UMAP ``n_neighbors``.

   ``umap.n_components``
      :Type: list of ``int``
      :Default: ``[5]``

      Candidate values for UMAP ``n_components`` (embedding dimensions
      passed to HDBSCAN).

   ``umap.metric``
      :Type: ``str``
      :Default: ``cosine``

      Distance metric used by UMAP.

   ``umap.min_dist``
      :Type: ``float``
      :Default: ``0.0``

      Controls how tightly UMAP packs points in the embedding.  ``0.0``
      is recommended for clustering.

   ``umap.low_memory``
      :Type: ``bool``
      :Default: ``false``

      Enable low-memory mode for very large corpora (slower).

   ``umap.random_state``
      :Type: ``int``
      :Default: ``42``

      Random seed for reproducibility.

``topic_model.bertopic``
   :Type: mapping
   :Default: all defaults applied

   ``bertopic.transformer_model``
      :Type: ``str``
      :Default: ``allenai/specter2_base``

      HuggingFace model identifier used to produce document embeddings.
      ``specter2_base`` is pre-trained on scientific text and is the
      recommended default for academic literature reviews.

   ``bertopic.n_gram_range``
      :Type: ``str``
      :Default: ``bigram``
      :Values: ``unigram`` | ``bigram``

      N-gram range for the c-TF-IDF vocabulary.

   ``bertopic.language``
      :Type: ``str``
      :Default: ``english``

   ``bertopic.calculate_probabilities``
      :Type: ``bool``
      :Default: ``true``

      Compute soft topic membership probabilities for each document.
      Required for topic distribution approximation.

``topic_model.berteley``
   :Type: mapping
   :Default: all defaults applied

   Pre-processing options for the Berteley text normaliser.

   ``berteley.allow_abbrev``
      :Type: ``bool``
      :Default: ``false``

      Allow abbreviation expansion during tokenisation.

``topic_model.ctfidf``
   :Type: mapping
   :Default: all defaults applied

   c-TF-IDF weighting options.

   ``ctfidf.bm25_weighting``
      :Type: ``bool``
      :Default: ``true``

      Apply BM25-style term weighting to c-TF-IDF.

   ``ctfidf.reduce_frequent_words``
      :Type: ``bool``
      :Default: ``true``

      Down-weight terms that appear frequently across many topics.

``topic_model.topic_distribution``
   :Type: mapping
   :Default: all defaults applied

   Parameters for the sliding-window topic distribution approximation.

   ``topic_distribution.window``
      :Type: ``int``
      :Default: ``8``

      Sliding-window size (in tokens) for distribution approximation.

   ``topic_distribution.stride``
      :Type: ``int``
      :Default: ``1``

      Window stride.

   ``topic_distribution.min_similarity``
      :Type: ``float``
      :Default: ``0.1``

      Minimum cosine similarity for a window to contribute to a topic's
      distribution.

   ``topic_distribution.batch_size``
      :Type: ``int``
      :Default: ``1000``

      Documents processed per batch during distribution approximation.

``topic_model.export`` *(topic_model)*
   :Type: mapping
   :Required: yes

   ``export.export_dir``
      :Type: ``str``
      :Required: yes

   ``export.run_name``
      :Type: ``str``
      :Default: ``null`` → auto-generated timestamp


----


``topic_report`` — PDF report generation
------------------------------------------

Selects one model configuration from the topic-model results and generates the
PDF report. Optionally uses the ``llm`` section for topic and community labels.

Which configuration is rendered is decided by ``topic_model.best_model_index``,
not here.

``topic_report.run_dir``
   :Type: ``str``
   :Default: ``null`` (auto-detect latest topic_model run)

   Path to a specific topic-model run directory.  Leave blank to use
   the most recent run in ``topic_model.export.export_dir``.

``topic_report.export_to``
   :Type: ``str``
   :Required: yes

   Directory where the generated PDF is written.  Interactive HTML versions of
   the network figures and the paper-selection annex are written here too.

``topic_report.meta``
   :Type: mapping
   :Default: all defaults applied

   Cover-page metadata.

   ``meta.title``
      :Type: ``str``
      :Default: ``Bibliographic report — Pysyrev``

   ``meta.subtitle``
      :Type: ``str``
      :Default: ``null``

   ``meta.author``
      :Type: ``str``
      :Default: ``Report generated with the pysyrev engine (v0.1)``

   ``meta.date_format``
      :Type: ``str``
      :Default: ``%d/%m/%Y``

      ``strftime``-compatible format string for the report date.

   ``meta.version``
      :Type: ``str``
      :Default: ``1.0.0``

   ``meta.summary``
      :Type: ``str``
      :Default: ``null``

      Optional introductory paragraph shown on the cover page.

``topic_report.sections``
   :Type: mapping
   :Default: all defaults applied

   Every section is rendered with its defaults unless you override it.  A
   section is skipped only when the data it needs is missing — the network
   panels, for instance, need a ``topic_model.doc_dataset`` carrying reference
   columns.

   ``sections.topics``

      ``topics.n_repr_docs_per_topic``
         :Type: ``int``
         :Default: ``5``

         Number of representative documents (closest to the topic
         centroid) displayed in the per-topic section.

   ``sections.bib_network``
      :Type: mapping
      :Default: all defaults applied

      Bibliographic coupling and co-citation panels, recomputed from the
      reviewed corpus' references at report time.  When the corpus carries a
      ``reference_keys`` column (see ``bib.reference_keys``) it is preferred
      over the raw ``references`` column, so coupling holds across merged
      sources.

      ``bib_network.coupling``

         ``coupling.resolution_range``
            :Type: list of ``float``
            :Default: ``[0.4, 1.2]``

            ``[min, max]`` bounds of the Leiden resolution sweep; the partition
            with the best modularity is kept.

         ``coupling.resolution_step``
            :Type: ``float``
            :Default: ``0.1``

         ``coupling.min_size``
            :Type: ``int``
            :Default: ``5``

            Communities smaller than this fall into the uncoupled tail.

         ``coupling.backbone_k``
            :Type: ``int``
            :Default: ``3``

            Draw only each node's *k* strongest couplings, so the figure stays
            readable on a dense corpus.

         ``coupling.color_by``
            :Type: ``str``
            :Default: ``topic``
            :Values: ``topic`` | ``community``

      ``bib_network.cocitation``

         Same keys as ``coupling`` (``resolution_range``, ``resolution_step``,
         ``min_size``, ``backbone_k``, ``color_by``, the last defaulting to
         ``community``), plus:

         ``cocitation.min_ref_freq``
            :Type: ``int``
            :Default: ``2``

            Keep only references cited by at least this many documents.  The
            once-cited long tail is dropped before anything is looked up.

         ``cocitation.metadata``
            :Type: mapping
            :Default: ``null`` (nodes left unresolved)

            Resolve co-citation nodes to real works, so communities can be read
            as intellectual bases (authors, venues, period) rather than opaque
            ids.  A failure here degrades to unnamed clusters; it never fails
            the report.

            ``metadata.provider``
               :Type: ``str``
               :Default: ``openalex``

            ``metadata.api_key`` / ``metadata.email``
               :Type: ``str``
               :Default: ``null``

               Typically ``${OPENALEX_API_KEY}``; the e-mail opts into the
               polite pool.

            ``metadata.cache``
               :Type: ``str``
               :Default: ``null``

               JSON cache path.  Resolved references are reused across runs.

            ``metadata.include_abstracts``
               :Type: ``bool``
               :Default: ``false``

      ``bib_network.connectivity``

         Inter-community and inter-topic connectivity matrices (mean coupling
         between groups).

         ``connectivity.scale``
            :Type: ``float``
            :Default: ``1000.0``

            Rescales the very small coupling values so the matrix is readable.

   ``sections.temporal``

      ``temporal.variants``
         :Type: list of ``str``
         :Default: ``[absolute, normalized, weighted]``
         :Values: any subset of ``absolute``, ``cumulative``, ``normalized``, ``weighted``

         Publication-trend chart variants included in the temporal
         analysis section.

   ``sections.topic_characteristics``

      ``topic_characteristics.n_top_cited_per_topic``
         :Type: ``int``
         :Default: ``5``

         Number of most-cited papers per topic used to compute citation
         impact scores.

      ``topic_characteristics.n_top_cited_global``
         :Type: ``int``
         :Default: ``50``

         Number of most-cited papers globally used to analyse topic
         distribution among highly cited documents.

   ``sections.topic_similarity``

      ``topic_similarity.clustering``
         :Type: ``bool``
         :Default: ``true``

         Reorder the similarity heatmap rows/columns by hierarchical
         clustering.

      ``topic_similarity.dendrogram``
         :Type: ``bool``
         :Default: ``true``

         Display a dendrogram alongside the heatmap.

   ``sections.paper_selection``

      The reading list: the papers worth reading first in each topic.

      ``paper_selection.min_year``
         :Type: ``int``
         :Default: ``2000``

         Only papers published from this year onward are eligible.

      ``paper_selection.proportion_per_topic``
         :Type: ``float``
         :Default: ``0.15``

         Fraction of each topic's documents included in the selection.

      ``paper_selection.selection_by``
         :Type: ``str``
         :Default: ``citations``
         :Values: ``citations`` | ``centrality`` | ``composite``

         How papers are ranked within a topic.  ``citations`` uses the raw
         citation count; ``centrality`` uses coupling-network centrality;
         ``composite`` combines three axes (see ``composite`` below) and needs
         the coupling network, i.e. a corpus carrying references.

      ``paper_selection.composite``
         :Type: mapping
         :Default: all defaults applied

         Parameters of the three-axis composite score — coupling centrality,
         thematic representativeness, and citation relevance.

         ``composite.aggregate``
            :Type: ``str``
            :Default: ``mean``
            :Values: ``mean`` | ``gmean`` | ``chebyshev``

            How the three axes are combined.  ``gmean`` and ``chebyshev``
            penalise a paper that is weak on any single axis.

         ``composite.weights``
            :Type: list of ``float``
            :Default: ``null`` (equal weights)

            ``(centrality, representativeness, relevance)``.

         ``composite.relevance_mode``
            :Type: ``str``
            :Default: ``blend``
            :Values: ``blend`` | ``cpy``

            ``blend`` mixes citations per year with raw citations, so an old
            landmark is not beaten by a recent paper with a good rate;
            ``cpy`` uses citations per year alone.

         ``composite.drop_current_year``
            :Type: ``bool``
            :Default: ``true``

            Drop the incomplete current year from the citations-per-year
            denominator.

         ``composite.current_year``
            :Type: ``int``
            :Default: ``null`` (latest year in the corpus)

         ``composite.split_current_year``
            :Type: ``bool``
            :Default: ``false``

            List current-year "research fronts" separately, ranked on two axes,
            instead of mixing them into the historical three-axis list.

      ``paper_selection.export_annex``
         :Type: ``bool``
         :Default: ``true``

         Write the full reference list of selected papers next to the PDF.

      ``paper_selection.annex_format``
         :Type: ``str``
         :Default: ``csv``
         :Values: ``csv`` | ``txt``

   ``sections.extra``
      :Type: list of mappings
      :Default: ``null``

      Extra sections appended verbatim to the report, in the declarative block
      format the PDF engine consumes (``paragraph``, ``table``, ``image``,
      ``callout``, …).


----


``llm`` — topic label generation
----------------------------------

When present, an LLM generates human-readable labels for each topic
discovered by the topic-model stage **and** for each bibliographic-coupling and
co-citation community of the report's network panels.  Used together with
``topic_report``.

Both label sets are cached under the run directory (``topic_labels/`` and
``cluster_labels/``), the community cache keyed on a hash of the sub-theme
terms themselves.  So the labeller runs once and re-runs only when the
partition actually changes — never on every report.

``llm.provider``
   :Type: ``str``
   :Required: yes
   :Values: ``anthropic`` | ``openai`` | ``albert`` | ``litellm`` | ``ollama``

``llm.model_id``
   :Type: ``str``
   :Required: yes

   Model identifier, e.g. ``claude-haiku-4-5-20251001``.

``llm.host``
   :Type: ``str``
   :Default: ``null`` (use default hosted endpoint)

   Custom endpoint for ``litellm`` or ``ollama``.

``llm.max_tokens``
   :Type: ``int``
   :Default: ``200``

``llm.temperature``
   :Type: ``float``
   :Default: ``0.3``

``llm.max_retries``
   :Type: ``int``
   :Default: ``2``

``llm.max_concurrent_requests``
   :Type: ``int``
   :Default: ``5``

``llm.requests_per_minute``
   :Type: ``int``
   :Default: ``null`` → provider default (Albert: 10)

   Shares the review's quota when both run against the same key.

``llm.timeout``
   :Type: ``float``
   :Default: ``null`` → provider default

   Seconds allowed per labelling call.

``llm.n_repr_docs_for_labeling``
   :Type: ``int``
   :Default: ``3``

   Number of representative documents sent to the LLM per label — the
   documents closest to a topic's centroid, or the most central papers of a
   bibliographic-coupling community.

``llm.system_prompt``
   :Type: ``str``
   :Default: ``null`` (built-in default prompt)

   Override the default system prompt for topic labelling.


----


``download`` — full-text retrieval (separate file)
----------------------------------------------------

``pysyrev download`` reads its own YAML file, not the pipeline config. Load it
with ``--config``; the positional arguments always override ``doc_dataset`` and
``output_dir``. Without ``--config``, only the OpenAlex step runs — no key
needed. See ``config_examples/download_config.yaml``.

Each paper is tried in cascade order: **Unpaywall** (legal open-access PDF by
DOI) → **OpenAlex** (the ``open_access.oa_url`` already in the input dataset)
→ **Elsevier TDM** (only when a key is set). Papers that could not be retrieved
are listed in ``download_report.csv`` with ``status = manual``.

``doc_dataset``
   :Type: ``str``
   :Required: yes

   CSV listing the papers to download; must contain a ``doi`` column.  The
   report's paper-selection annex and ``reviewed_included.csv`` both qualify.

``output_dir``
   :Type: ``str``
   :Required: yes

   Root for the output: ``papers/`` holds the files, ``download_report.csv``
   holds one row per paper (doi | title | status | source | file_path).

``format``
   :Type: ``str``
   :Default: ``pdf``
   :Values: ``pdf`` | ``xml``

   Default format for the open-access sources.  ``xml`` gives structured
   full text — smaller, and already sectioned, when the next step is an LLM.

``max_papers``
   :Type: ``int``
   :Default: ``null`` (all)

``request_delay``
   :Type: ``float``
   :Default: ``1.0``

   Seconds between HTTP requests.  Unpaywall's fair-use policy requires ≤ 1
   request per second.

``unpaywall``
   :Type: mapping
   :Default: ``null`` (step skipped)

   ``unpaywall.email``
      :Type: ``str``
      :Required: yes

      Required by Unpaywall's politeness policy; unlocks the
      higher-throughput pool.

``elsevier``
   :Type: mapping
   :Default: ``null`` (step skipped)

   ``elsevier.api_key``
      :Type: ``str``
      :Required: yes

      Institutional Elsevier TDM key.

   ``elsevier.format``
      :Type: ``str``
      :Default: ``xml``
      :Values: ``xml`` | ``pdf``
