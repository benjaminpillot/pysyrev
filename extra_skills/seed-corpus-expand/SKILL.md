---
name: seed-corpus-expand
description: Turn a set of seed papers into a deduplicated, prefiltered candidate pool via OpenAlex — the seed→pool stage of a literature map, upstream of scope-screening. Unions the seeds with forward citations (cites:), backward citations (referenced_works), and title/abstract phrase searches; dedups by OpenAlex id AND normalised title key (catching OpenAlex's duplicate records with empty DOIs); applies a year/type/retraction/language prefilter tolerant of OpenAlex's empty language field. Returns the pool to LLM scope-screen — screening is deliberately left out because it depends on the subfield's scope definition. Companion upstream of corpus-map's clustering stage.
---

# seed-corpus-expand — seeds → candidate pool

The reproducible front half of building a screened corpus: given a handful of
**seed papers** squarely in a subfield, assemble the pool of candidates to
screen. One call, `expand_from_seeds(...)`, runs the whole expansion; the
individual stages are also exposed if you want to inspect or reorder them.

This is the stage **upstream** of `corpus-map`'s clustering. It stops at the
*pool* on purpose — the next step, LLM scope-screening against a written SCOPE
string, is subfield-specific and belongs to your judgement, not to a generic
helper (see `corpus-map` step 5).

## Requires

An OpenAlex API key: the `OPENALEX_API_KEY` env var is read automatically, or
pass `api_key=`. Runs in a python env with `requests` (OpenAlex is allowlisted).

## The pipeline (what `expand_from_seeds` does)

```python
res = expand_from_seeds(
    seed_ids=["10.1038/s41467-020-18661-9", "W4402888209", ...],  # DOIs or Wxxxx
    api_key=os.environ["OPENALEX_API_KEY"],
    phrases=["critical minerals energy transition",
             "material footprint renewable power", ...],   # 10–25 targeted queries
    year_min=2022, year_max=2026,
)
res["counts"]     # {'raw_union':…, 'deduped':…, 'prefiltered':…}
res["pool"]       # list of candidate records (dicts) to scope-screen
res["seed_ids"]   # set of resolved seed ids — flag is_seed on pool members
```

The union has four sources, each a helper you can call alone:

1. **Seeds themselves** — always candidates (`fetch_records`).
2. **Forward citations** — papers that cite the seeds, `cites:<id>`
   (`forward_citations`). DOIs are resolved to Wxxxx ids first, since `cites:`
   needs the OpenAlex id.
3. **Backward citations** — the seeds' own `referenced_works`
   (`backward_citations`).
4. **Phrase searches** — union of `title_and_abstract.search` over your query
   list (`search_phrases`), year-bounded.

Then **`dedup_pool`** collapses duplicates by OpenAlex id AND by normalised
title key — OpenAlex issues duplicate records, sometimes with an empty DOI on
one copy, so an id-only dedup leaves doubles; the title key catches them, and
the copy with a DOI / longer abstract is kept. Finally **`passes_prefilter`**
applies year window, allowed types, retraction flag, and a language filter that
is **tolerant of OpenAlex's empty language field** (empty passes; only an
explicit non-`en` code excludes — a real robustness fix, empty language is
common on in-scope conference papers).

## Choosing seeds and phrases

- **Seeds**: 5–15 papers unambiguously in the subfield, ideally spanning its
  sub-themes so forward/backward expansion reaches every corner. Anchor seeds
  outside the year window still work — they seed the expansion even if the
  prefilter later drops them.
- **Phrases**: 10–25 short queries in the field's own vocabulary. These reach
  papers that neither cite nor are cited by the seeds but use the same terms —
  the recall backstop for a young or fragmented subfield.

## After the pool

1. Reconstruct abstracts with `unabstract(record["abstract_inverted_index"])`.
2. Split into *has-abstract* / *no-abstract* streams (`has_screenable_abstract`).
3. LLM-screen each against your SCOPE string; no-abstract records screen on
   title/venue/topics or get completed from Web of Science first — this is
   `corpus-map` step 5–6.
4. Cluster the confirmed corpus by bibliographic coupling — `corpus-map`.

## Tuning & pitfalls

- `max_per_seed` / `max_per_phrase` cap each source; raise for exhaustive
  recall, lower for a quick draft. Forward citations can be large for a
  highly-cited seed.
- `use_forward` / `use_backward` toggle the citation arms — a phrase-only pool
  is a keyword corpus (no citation expansion), useful as a comparison baseline.
- The pool is **candidates, not corpus**. Do not skip screening: raw expansion
  pulls in broad "energy transition" papers well outside a narrow material
  scope (on this project, a 689-raw expansion screened down to ~21 in-scope).
- `select=` controls fetched fields; the default carries everything the
  downstream clustering needs (`referenced_works`, `abstract_inverted_index`,
  `topics`, `primary_location`).
