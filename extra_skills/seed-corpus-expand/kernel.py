"""seed-corpus-expand: turn a set of seed papers into a deduplicated,
prefiltered candidate pool via OpenAlex — the seed -> pool half of a corpus
map, upstream of LLM scope-screening. See SKILL.md.

Requires an OpenAlex API key (OPENALEX_API_KEY env var, or pass api_key=).
Networking via `requests`; OpenAlex is allowlisted in the analysis kernel.
"""
import re
import time

BASE = "https://api.openalex.org/works"
SELECT = ("id,doi,title,display_name,publication_year,type,language,"
          "cited_by_count,referenced_works,abstract_inverted_index,"
          "primary_location,topics")


def normalize_id(x):
    """Accept an OpenAlex id (W..., or full URL) or a DOI; return the token
    usable in an openalex_id / doi filter."""
    x = (x or "").strip()
    if x.startswith("http") and "openalex.org/" in x:
        return x.rsplit("/", 1)[-1]
    if x.lower().startswith("10.") or "doi.org" in x:
        return "doi:" + x.replace("https://doi.org/", "").strip()
    return x  # already a Wxxxx id


def unabstract(inverted_index):
    """OpenAlex abstract_inverted_index -> plain text. Empty on missing."""
    if not inverted_index:
        return ""
    pos = {p: tok for tok, ps in inverted_index.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def title_key(title):
    """Normalised key for dedup across duplicate OpenAlex records."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:90]


def passes_prefilter(record, year_min, year_max=None,
                     types=None, require_english=True):
    """Year/type/retraction/language prefilter, tolerant of empty language.
    OpenAlex leaves `language` empty on many in-scope records — never exclude
    on empty language; only an explicit non-'en' code excludes."""
    if types is None:
        types = ("article", "review", "preprint", "book-chapter",
                 "conference-paper")
    if record.get("is_retracted"):
        return False
    y = record.get("publication_year")
    if y is None or y < year_min:
        return False
    if year_max is not None and y > year_max:
        return False
    if record.get("type") not in types:
        return False
    if require_english:
        lang = (record.get("language") or "").strip().lower()
        if lang and lang != "en":
            return False
    return True


def oa_get(params, api_key, tries=4, pause=0.4):
    """One OpenAlex GET with retry on transient proxy/5xx errors."""
    import requests
    params = dict(params); params["api_key"] = api_key
    last = None
    for t in range(tries):
        try:
            r = requests.get(BASE, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = type(e).__name__
        time.sleep(pause * (t + 1))
    raise RuntimeError(f"OpenAlex request failed after {tries} tries: {last}")


def fetch_records(ids, api_key, select=None, chunk=90):
    """Fetch full records for a list of OpenAlex ids / DOIs. Returns
    {openalex_id: record}. Batches via the openalex_id|... filter."""
    select = select or SELECT
    ids = [normalize_id(x) for x in ids if x]
    wids = [i for i in ids if not i.startswith("doi:")]
    dois = [i for i in ids if i.startswith("doi:")]
    out = {}
    for group, key in ((wids, "openalex_id"), (dois, "doi")):
        vals = [g.split(":", 1)[1] if key == "doi" else g for g in group]
        for i in range(0, len(vals), chunk):
            batch = vals[i:i + chunk]
            filt = f"{key}:" + "|".join(batch)
            data = oa_get({"filter": filt, "select": select, "per-page": chunk},
                        api_key)
            for w in data.get("results", []):
                out[w["id"]] = w
    return out


def oa_paged(filt, api_key, select, max_records, per_page=200):
    """Cursor-paged pull of a filter, capped at max_records."""
    out = []
    cursor = "*"
    while cursor and len(out) < max_records:
        data = oa_get({"filter": filt, "select": select,
                     "per-page": per_page, "cursor": cursor}, api_key)
        out.extend(data.get("results", []))
        cursor = (data.get("meta") or {}).get("next_cursor")
    return out[:max_records]


def forward_citations(seed_ids, api_key, select=None, max_per_seed=600):
    """Papers that CITE the seeds (cites:<id>). Union across seeds."""
    select = select or SELECT
    seen = {}
    for sid in seed_ids:
        wid = normalize_id(sid)
        if wid.startswith("doi:"):        # need the Wxxxx id to use cites:
            r = fetch_records([wid], api_key, select="id")
            wid = next(iter(r), None)
            if not wid:
                continue
            wid = wid.rsplit("/", 1)[-1]
        for w in oa_paged(f"cites:{wid}", api_key, select, max_per_seed):
            seen[w["id"]] = w
    return seen


def backward_citations(seed_records, api_key, select=None):
    """Papers the seeds cite (their referenced_works). seed_records is the
    dict returned by fetch_records(seeds)."""
    select = select or SELECT
    refs = set()
    for r in seed_records.values():
        refs.update(r.get("referenced_works") or [])
    return fetch_records(list(refs), api_key, select=select)


def search_phrases(phrases, api_key, year_min=None, year_max=None,
                   select=None, max_per_phrase=400):
    """Union of title_and_abstract.search over a list of phrase queries."""
    select = select or SELECT
    seen = {}
    for ph in phrases:
        filt = f"title_and_abstract.search:{ph}"
        if year_min:
            filt += f",from_publication_date:{year_min}-01-01"
        if year_max:
            filt += f",to_publication_date:{year_max}-12-31"
        for w in oa_paged(filt, api_key, select, max_per_phrase):
            seen[w["id"]] = w
    return seen


def dedup_pool(records):
    """Deduplicate a dict/list of records by OpenAlex id AND normalised title
    key. OpenAlex emits duplicate records, sometimes with an empty DOI on one
    copy — the title key catches those. Keeps the copy with the richer record
    (has DOI, then longer abstract). Returns a list of records."""
    recs = list(records.values()) if isinstance(records, dict) else list(records)
    by_key = {}
    for r in recs:
        k = title_key(r.get("title") or r.get("display_name"))
        k = k or r["id"]                       # untitled: fall back to id
        keep = by_key.get(k)
        if keep is None:
            by_key[k] = r; continue
        # prefer the record with a DOI, then the longer abstract
        score = (bool(r.get("doi")),
                 len(r.get("abstract_inverted_index") or {}))
        kscore = (bool(keep.get("doi")),
                  len(keep.get("abstract_inverted_index") or {}))
        if score > kscore:
            by_key[k] = r
    return list(by_key.values())


def expand_from_seeds(seed_ids, api_key, phrases=None,
                      year_min=None, year_max=None,
                      types=None, require_english=True,
                      use_forward=True, use_backward=True,
                      select=None):
    """END-TO-END seed -> deduplicated, prefiltered candidate pool.

    Unions: the seeds themselves + forward citations (cites:) + backward
    citations (referenced_works) + title/abstract phrase searches; dedups by
    id and title key; applies the year/type/language prefilter.

    Returns dict:
      seeds        : {id: record} for the resolved seeds
      pool         : list of prefiltered, deduplicated candidate records
      seed_ids     : set of resolved seed OpenAlex ids (to flag is_seed)
      counts       : per-stage sizes (raw union -> deduped -> prefiltered)

    NOTE: this is the pool to SCOPE-SCREEN (an LLM classifier against a written
    SCOPE string) — it is NOT the final corpus. Screening is deliberately left
    out: it depends on your subfield's scope definition, not on this mechanics.
    """
    select = select or SELECT
    phrases = phrases or []
    seeds = fetch_records(seed_ids, api_key, select=select)
    union = dict(seeds)                        # seeds are always candidates
    if use_forward:
        union.update(forward_citations(seed_ids, api_key, select=select))
    if use_backward:
        union.update(backward_citations(seeds, api_key, select=select))
    if phrases:
        union.update(search_phrases(phrases, api_key, year_min, year_max,
                                    select=select))
    raw = len(union)
    deduped = dedup_pool(union)
    pool = [r for r in deduped
            if passes_prefilter(r, year_min or 0, year_max, types,
                                require_english)]
    return {
        "seeds": seeds,
        "seed_ids": set(seeds.keys()),
        "pool": pool,
        "counts": {"raw_union": raw, "deduped": len(deduped),
                   "prefiltered": len(pool)},
    }
