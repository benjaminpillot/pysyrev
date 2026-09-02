---
name: corpus-map
description: Build a bibliometric map of a research subfield from seed papers — expand via OpenAlex citations and keyword queries, screen each candidate against an LLM scope definition, cluster by bibliographic coupling, and label clusters. Use when a user wants to "map the literature", survey a subfield, find its structure/clusters/trends, or build a screened paper corpus from seeds. Encodes two robustness fixes: OpenAlex leaves the language field empty on many records (never filter with lang == "en"), and records with no abstract must be screened on title or completed from Web of Science before exclusion.
---

# corpus-map

Build a clustered literature map of a subfield: seeds -> expansion -> LLM
scope screening -> bibliographic-coupling clusters -> labels + figure.

Runs in a python env with `openalex` reachable (allowlisted), plus
`scikit-learn`, `python-igraph`, `leidenalg`, `matplotlib`. The OpenAlex key
is in `OPENALEX_API_KEY`. `kernel.py` ships the load-bearing helpers; call
them rather than re-deriving.

## Why this skill exists

Two failure modes cost real papers on the first build of this pipeline, and
both are silent — they drop in-scope work with no error:

1. **OpenAlex leaves `language` empty on many records** (~5% of a pool,
   heavily among conference papers and non-English-metadata venues). An
   equality test `record["language"] == "en"` drops every one of them.
   **Never** filter on language equality. `passes_prefilter()` treats empty
   language as acceptable; only an explicit non-English code excludes.
2. **Records with no abstract cannot be LLM-screened on abstract.** OpenAlex
   (and Crossref) carry no abstract for a nontrivial share of records,
   including some highly-cited ones. Dropping them silently biases the corpus.
   Screen them on **title + venue + topics** instead, or complete the abstract
   from Web of Science (see `wos_complete()`) before screening.

## Workflow

1. **Seeds.** Collect 5-15 DOIs/OpenAlex IDs squarely in the subfield. Fetch
   them with `select=` including `referenced_works,abstract_inverted_index`.
2. **Expand.** Union of: forward citations (`cites:<id>`), backward citations
   (each seed's `referenced_works`), and 10-25 targeted
   `title_and_abstract.search` phrase queries. Dedup by OpenAlex id AND by a
   normalised title key (`title_key()`) — OpenAlex issues duplicate records,
   sometimes with an empty DOI on one copy.
3. **Reconstruct abstracts** with `unabstract(record["abstract_inverted_index"])`.
4. **Prefilter** with `passes_prefilter()` — year window, type, retraction,
   language-tolerant. Split the survivors into *has-abstract* and
   *no-abstract* streams.
5. **Screen** each has-abstract candidate against a written `SCOPE` string
   with `host.llm(..., system=SCOPE)` (fan out with `max_concurrency`). Screen
   the no-abstract stream on title/venue/topics with the same SCOPE. Verdict =
   text starts with "IN".
6. **(Optional) WoS completion.** If a Web of Science Expanded key is present,
   `wos_complete(dois, key)` returns abstracts (and cited-reference DOIs) for
   records OpenAlex left bare; re-screen those on the recovered abstract. WoS
   under-indexes preprints and many conference series — use it to *complete*,
   never as the primary search.
7. **Saturate.** Snowball from confirmed papers and re-screen a blind sample
   of rejects (audit prompt: "was this exclusion correct?"). A low
   false-negative rate means the corpus ceiling is the field's real size.
8. **Cluster.** `salton_coupling()` -> igraph -> Leiden
   (`RBConfigurationVertexPartition`, sweep `resolution_parameter` 0.4-0.9,
   report modularity). Papers with no coupled partner are genuinely
   unstructured — keep them, report the count, exclude from the layout.
9. **Label** clusters from distinguishing TF-IDF terms (`cluster_terms()`) and
   central papers by within-cluster PageRank. Quantify cross-coupling
   (within vs between) to state how separated the clusters are.

## Scope definition (step 5)

The single biggest quality lever is the SCOPE string. Enumerate BOTH what is
in scope and what is out, in the field's own vocabulary, with the boundary
cases that keyword filters get wrong. Keyword screening cannot draw a semantic
boundary — an LLM classifier against an explicit definition can. Spot-check ~15
accepted and ~15 rejected before trusting it, and re-audit a reject sample.

## Deliverables

Four-panel figure (coupling network, corpus-size comparison, cluster-trajectory
lines, theme-coverage bars), a corpus CSV (one row per paper: cluster, coupling
degree, abstract_source, is_seed), a cluster-summary CSV, and a written
synthesis. Verify every cited DOI resolves (Crossref) and carries no retraction
flag before delivery. Flag partial final years in trajectory panels.
