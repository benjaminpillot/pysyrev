"""
Actors of a literature — who writes it, from where.

A corpus row records three kinds of *actors*: its authors, their institutions,
and those institutions' countries. Every source knows them, each in its own
shape (OpenAlex ``authorships`` with ORCID/ROR, WoS address blocks and
organisation-enhanced names, Scopus/PubMed free-text addresses), and none of
those shapes survives the merge into the flat ``author`` / ``affiliations``
strings. This module keeps them in one source-agnostic column, :data:`ACTORS`,
so that the rankings and collaboration networks built on top of it
(:mod:`pysyrev.core.networks.collaboration`) never look at a source format.

The column holds, per paper, a JSON object of *paper-level* sets::

    {"authors":      [{"key": ..., "name": ...}, ...],
     "institutions": [{"key": ..., "name": ..., "country": "FR" | null}, ...],
     "countries":    ["FR", "SN", ...]}

Paper level is deliberate: it is the one granularity every source delivers
reliably (the OpenAlex CSV export, for one, flattens which author belongs to
which institution), and it is all the analyses need — an actor is *involved* in
a paper, and two actors *collaborate* when they share one.

Keys carry their identity scheme, strongest first:

    authors       ``orcid:`` > ``openalex:`` / ``wos:`` (source person id) > ``name:``
    institutions  ``ror:``   > ``openalex:``                               > ``name:``
    countries     ISO 3166-1 alpha-2 codes

``orcid:`` and ``ror:`` are shared across sources, so OpenAlex and WoS records
of the same person or institution meet on them directly. ``name:`` keys (a
normalised name, the only identity some sources give) are reconciled with the
identified ones afterwards by :func:`harmonize_actors`.

Layout: the generic core first (schema, keys, country and address helpers,
the address-based fallback, harmonisation), then one adapter per source.
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

#: Name of the canonical actors column (see the module docstring).
ACTORS: str = "actors"

#: Actor levels, in the order the analyses present them.
LEVELS = ("author", "institution", "country")

# Plural JSON field of each level.
_FIELD = {"author": "authors", "institution": "institutions", "country": "countries"}


# =============================================================================
# Schema
# =============================================================================

def empty_actors() -> dict:
    return {"authors": [], "institutions": [], "countries": []}


def _dedupe(entries: Iterable[dict]) -> List[dict]:
    """Keep the first entry of each key, in order; drop key-less entries.

    A later duplicate still fills a field the first one left empty (e.g. the
    country of an institution first seen without one).
    """
    out: Dict[str, dict] = {}
    for e in entries:
        if not e or not e.get("key"):
            continue
        kept = out.get(e["key"])
        if kept is None:
            out[e["key"]] = dict(e)
        else:
            for k, v in e.items():
                if v and not kept.get(k):
                    kept[k] = v
    return list(out.values())


def make_actors(authors: Iterable[dict] = (),
                institutions: Iterable[dict] = (),
                countries: Iterable[Optional[str]] = ()) -> Optional[str]:
    """Encode one paper's actors as the :data:`ACTORS` JSON cell.

    *authors* are ``{"key", "name"}`` dicts, *institutions* ``{"key", "name",
    "country"}`` dicts, *countries* ISO2 codes (``None`` entries are skipped).
    Each list is deduplicated by key; the paper's countries are the union of
    *countries* and its institutions' countries. Returns ``None`` when the
    paper has no actor at all, so the cell reads as missing like any other.
    """
    authors = _dedupe(authors)
    institutions = _dedupe(institutions)
    seen, codes = set(), []
    for c in list(countries) + [i.get("country") for i in institutions]:
        if c and c not in seen:
            seen.add(c)
            codes.append(c)
    if not (authors or institutions or codes):
        return None
    return json.dumps({"authors": authors, "institutions": institutions,
                       "countries": codes}, ensure_ascii=False)


def parse_actors(cell) -> dict:
    """Decode an :data:`ACTORS` cell; a missing or malformed one is empty."""
    if isinstance(cell, dict):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return empty_actors()
    try:
        data = json.loads(cell)
    except ValueError:
        return empty_actors()
    out = empty_actors()
    out.update({k: data.get(k) or [] for k in out})
    return out


def level_entities(cell, level: str) -> List[dict]:
    """The entities of one *level* in a cell, as ``{"key", "name", ...}`` dicts.

    Countries are stored as bare codes; they come back as
    ``{"key": code, "name": country name}`` so every level reads the same.
    """
    actors = parse_actors(cell)
    if level == "country":
        return [{"key": c, "name": country_name(c)} for c in actors["countries"]]
    return actors[_FIELD[level]]


# =============================================================================
# Normalisation & keys
# =============================================================================

def _ascii(text: str) -> str:
    return (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore").decode("ascii"))


_CORRESPONDING_RE = re.compile(r"\(\s*corresponding author\s*\)", re.I)
_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?-?){1,4}$")


def _split_person_name(name: str):
    """``(last, first)`` from ``"Last, First"``, ``"First Last"`` or
    ``"Last F.M."`` (the Scopus form, initials after the surname)."""
    name = _CORRESPONDING_RE.sub("", name).strip()
    if "," in name:
        last, first = name.split(",", 1)
        return last.strip(), first.strip()
    tokens = name.split()
    if not tokens:
        return "", ""
    if len(tokens) > 1 and _INITIALS_RE.match(tokens[-1]):
        return " ".join(tokens[:-1]), tokens[-1]
    return tokens[-1], " ".join(tokens[:-1])


def _norm_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _ascii(text).lower()).split())


def person_name_norm(name, initials: bool = True) -> Optional[str]:
    """Normalised person name: ``"pires m"`` (*initials*) or ``"pires marcelo a"``.

    Surname plus the first initial is the classic bibliometric author key
    (bibliometrix, VOSviewer): it is the only form every source can produce,
    WoS standard names being ``"Pires, MA"``. The full form is used where both
    sides carry a first name, to reconcile identities more safely.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    last, first = _split_person_name(name)
    last, first = _norm_words(last), _norm_words(first)
    if not last:
        return None
    if not first:
        return last
    return f"{last} {first[0]}" if initials else f"{last} {first}"


_PARENS_RE = re.compile(r"\([^)]*\)")


def institution_norm(name) -> Optional[str]:
    """Normalised institution name for name-based identity.

    Accents, case, punctuation, a parenthesised acronym and a leading article
    are dropped and ``&`` is read as ``and``, so WoS' ``"Korea Advanced
    Institute of Science & Technology (KAIST)"`` and OpenAlex' ``"Korea Advanced
    Institute of Science and Technology"`` meet.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    text = _PARENS_RE.sub(" ", name.replace("\\&", "&").replace("&", " and "))
    text = _norm_words(text)
    text = re.sub(r"^the\s+", "", text)
    return text or None


def _strip_url(value: str, *prefixes: str) -> str:
    value = value.strip()
    for p in prefixes:
        if value.lower().startswith(p):
            return value[len(p):]
    return value


def orcid_key(value) -> Optional[str]:
    """``orcid:0000-0002-8333-7236`` from a bare ORCID or an orcid.org URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    oid = _strip_url(value, "https://orcid.org/", "http://orcid.org/").upper()
    return f"orcid:{oid}" if re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", oid) else None


def ror_key(value) -> Optional[str]:
    """``ror:02wnmk332`` from a bare ROR id or a ror.org URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    rid = _strip_url(value, "https://ror.org/", "http://ror.org/").lower()
    return f"ror:{rid}" if rid else None


def openalex_key(value) -> Optional[str]:
    """``openalex:A5033832888`` from an OpenAlex id or URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    oid = _strip_url(value, "https://openalex.org/", "http://openalex.org/")
    return f"openalex:{oid}" if oid else None


def author_name_key(name) -> Optional[str]:
    norm = person_name_norm(name)
    return f"name:{norm}" if norm else None


def institution_name_key(name) -> Optional[str]:
    norm = institution_norm(name)
    return f"name:{norm}" if norm else None


# =============================================================================
# Countries
# =============================================================================

# Country spellings pycountry does not know: WoS' abbreviated forms, the UK
# nations WoS reports separately, and common short names. Lower-case keys.
COUNTRY_ALIASES: Dict[str, str] = {
    "peoples r china": "CN", "china": "CN", "pr china": "CN", "p r china": "CN",
    "england": "GB", "scotland": "GB", "wales": "GB", "north ireland": "GB",
    "northern ireland": "GB", "uk": "GB", "u k": "GB", "great britain": "GB",
    "usa": "US", "u s a": "US", "united states of america": "US",
    "russia": "RU", "turkey": "TR", "turkiye": "TR",
    "u arab emirates": "AE", "uae": "AE",
    "dem rep congo": "CD", "democratic republic of the congo": "CD",
    "dr congo": "CD", "rep congo": "CG", "congo": "CG",
    "cote ivoire": "CI", "cote d ivoire": "CI", "ivory coast": "CI",
    "bosnia and herceg": "BA", "bosnia and herzegovina": "BA",
    "papua n guinea": "PG", "trinid and tobago": "TT",
    "cent afr republ": "CF", "central african republic": "CF",
    "dominican rep": "DO", "equat guinea": "GQ", "guinea bissau": "GW",
    "fr polynesia": "PF", "st kitts and nevi": "KN", "st vincent": "VC",
    "st lucia": "LC", "sao tome and prin": "ST", "turks and caicos": "TC",
    "antigua and barbu": "AG", "marshall island": "MH", "falkland island": "FK",
    "brunei": "BN", "byelarus": "BY", "macedonia": "MK", "north macedonia": "MK",
    "swaziland": "SZ", "cape verde": "CV", "micronesia": "FM",
    "vatican": "VA", "palestine": "PS", "south korea": "KR",
    "north korea": "KP", "korea": "KR", "republic of korea": "KR",
    "laos": "LA", "vietnam": "VN", "syria": "SY", "iran": "IR",
    "venezuela": "VE", "bolivia": "BO", "tanzania": "TZ", "moldova": "MD",
    "czech republic": "CZ", "czechia": "CZ", "the netherlands": "NL",
    "holland": "NL", "hong kong": "HK", "macau": "MO", "macao": "MO",
    "taiwan": "TW", "kosovo": "XK", "the gambia": "GM", "gambia": "GM",
    "bahamas": "BS", "reunion": "RE", "east timor": "TL",
}

# Countries reported under their sovereign's name (or no longer existing)
# that pycountry does not resolve — kept explicit so they are not guessed.
_COUNTRY_NAMES_EXTRA = {"XK": "Kosovo"}


@lru_cache(maxsize=4096)
def country_code(text) -> Optional[str]:
    """ISO 3166-1 alpha-2 code of a country name or code, or ``None``.

    Accepts ISO2/ISO3 codes, official and common names (via pycountry) and the
    WoS abbreviations in :data:`COUNTRY_ALIASES`. Never guesses: an unknown
    string is ``None``, not the closest match.
    """
    import pycountry

    if not isinstance(text, str):
        return None
    raw = text.strip().strip(".").strip()
    if not raw:
        return None
    if len(raw) == 2 and raw.isalpha():
        code = raw.upper()
        if code in _COUNTRY_NAMES_EXTRA or pycountry.countries.get(alpha_2=code):
            return code
    key = _norm_words(raw.replace("&", " and "))
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    try:
        return pycountry.countries.lookup(raw).alpha_2
    except LookupError:
        pass
    try:
        return pycountry.countries.lookup(_ascii(raw)).alpha_2
    except LookupError:
        return None


@lru_cache(maxsize=512)
def country_name(code) -> Optional[str]:
    """Display name of an ISO2 code (common name when pycountry has one)."""
    import pycountry

    if not isinstance(code, str):
        return None
    if code in _COUNTRY_NAMES_EXTRA:
        return _COUNTRY_NAMES_EXTRA[code]
    c = pycountry.countries.get(alpha_2=code)
    if c is None:
        return code
    return getattr(c, "common_name", None) or c.name


def _address_segments(address: str) -> List[str]:
    address = address.replace("\\&", "&")
    return [s.strip(" .;[]") for s in address.split(",") if s.strip(" .;[]")]


def country_from_address(address) -> Optional[str]:
    """Country of a free-text address — its last segment, or that segment's
    trailing words (``"Boston, MA 02155 USA"``, ``"Daejeon 34141, South Korea"``)."""
    if not isinstance(address, str):
        return None
    segments = _address_segments(address)
    if not segments:
        return None
    last = segments[-1]
    code = country_code(last)
    if code:
        return code
    words = re.sub(r"[\d-]+", " ", last).split()
    for n in (3, 2, 1):
        if len(words) >= n:
            code = country_code(" ".join(words[-n:]))
            if code:
                return code
    return None


# An address segment naming an organisation, strongest first: the university
# (or school-level body) before an institute or centre it may host.
_INST_TIER1 = re.compile(
    r"\b(univ|universit|universidad|universidade|universita|universite|"
    r"college|coll|ecole|polytech|politecn|eth|epfl|kth)", re.I)
_INST_TIER2 = re.compile(
    r"\b(inst|institut|academy|acad|ctr|centre|center|centro|lab|laborat|"
    r"hosp|hospital|agency|ministry|minist|council|cnrs|ird|inra|inrae|cirad|"
    r"csic|cnr|foundation|fdn|museum|observ|corp|inc|ltd|gmbh|company|bank|"
    r"organi[sz]ation|org\b|commission|res\b|research)", re.I)


def institution_from_address(address) -> Optional[str]:
    """Best guess of the organisation named in a free-text address.

    The first segment that names a university-level body, else the first that
    names an institute/centre/agency…, else the first segment. Used only when
    a source gives no structured institution.
    """
    if not isinstance(address, str):
        return None
    segments = _address_segments(address)
    if len(segments) > 1:
        segments = segments[:-1]            # the country
    segments = [s for s in segments if not re.fullmatch(r"[\d\s-]+", s)]
    if not segments:
        return None
    for pattern in (_INST_TIER1, _INST_TIER2):
        for s in segments:
            if pattern.search(s):
                return s
    return segments[0]


# =============================================================================
# Generic fallback: names + free-text addresses
# =============================================================================

def actors_from_addresses(author_names: Iterable[str],
                          addresses: Iterable[str]) -> Optional[str]:
    """Actors from a name list and free-text addresses — any source.

    Authors get ``name:`` keys; each address yields an institution (see
    :func:`institution_from_address`) and a country. The fallback for sources
    with no structured affiliation (Scopus, PubMed) and for the records a
    structured source leaves without one.
    """
    authors = [{"key": author_name_key(n), "name": n.strip()}
               for n in author_names if isinstance(n, str) and n.strip()]
    institutions = []
    for addr in addresses:
        name = institution_from_address(addr)
        if name:
            institutions.append({"key": institution_name_key(name), "name": name,
                                 "country": country_from_address(addr)})
    countries = [country_from_address(a) for a in addresses]
    return make_actors(authors, institutions, countries)


def split_authors(value, sep: Optional[str] = None) -> List[str]:
    """Split an ``author`` cell (``' and '``, ``'; '`` or ``'|'`` separated)."""
    if not isinstance(value, str) or not value.strip():
        return []
    if sep is None:
        sep = (" and " if " and " in value
               else "|" if "|" in value else ";")
    return [a.strip() for a in value.split(sep) if a.strip()]


def split_addresses(value) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [a.strip() for a in value.split(";") if a.strip()]


# =============================================================================
# Harmonisation (post-merge, source-agnostic)
# =============================================================================

def _alias_map(entities: Iterable[dict], norm) -> Dict[str, str]:
    """``{name_key: id_key}`` for name-keyed entities whose normalised name
    belongs to exactly one identified (non-``name:``) entity in the corpus."""
    ids_by_norm: Dict[str, set] = defaultdict(set)
    name_keys_by_norm: Dict[str, set] = defaultdict(set)
    for e in entities:
        n = norm(e.get("name"))
        if not n:
            continue
        if e["key"].startswith("name:"):
            name_keys_by_norm[n].add(e["key"])
        else:
            ids_by_norm[n].add(e["key"])
    alias = {}
    for n, name_keys in name_keys_by_norm.items():
        ids = ids_by_norm.get(n)
        if ids and len(ids) == 1:
            target = next(iter(ids))
            for k in name_keys:
                alias[k] = target
    return alias


def harmonize_actors(df: pd.DataFrame, col: str = ACTORS) -> pd.DataFrame:
    """Reconcile ``name:`` keys with the identified ones, across the corpus.

    After a merge, the same person may be ``orcid:…`` on an OpenAlex record and
    ``name:kim h`` on a WoS-only one. A name-keyed author or institution is
    re-keyed onto an identified one when their normalised names match and that
    name belongs to **one** identified entity only — an ambiguous name
    (two ORCIDs share it) is left alone rather than merged wrongly. Authors
    are matched on surname + full first name, then on surname + initial for
    the name-keyed entries that carry initials only. Idempotent; returns *df*
    with *col* rewritten (a no-op without the column).
    """
    if col not in df.columns:
        return df
    parsed = [parse_actors(v) for v in df[col]]
    authors = [a for p in parsed for a in p["authors"]]
    institutions = [i for p in parsed for i in p["institutions"]]

    author_alias = _alias_map(authors, lambda n: person_name_norm(n, initials=False))
    initial_alias = _alias_map(authors, person_name_norm)
    for k, v in initial_alias.items():
        author_alias.setdefault(k, v)
    inst_alias = _alias_map(institutions, institution_norm)
    if not (author_alias or inst_alias):
        return df

    def rekey(entries, alias):
        return [dict(e, key=alias.get(e["key"], e["key"])) for e in entries]

    out = []
    for raw, p in zip(df[col], parsed):
        if not (p["authors"] or p["institutions"] or p["countries"]):
            out.append(raw)
            continue
        out.append(make_actors(rekey(p["authors"], author_alias),
                               rekey(p["institutions"], inst_alias),
                               p["countries"]))
    df = df.copy()
    df[col] = out
    return df


# =============================================================================
# Source adapters
# =============================================================================

# ---- OpenAlex (API record) ---------------------------------------------------

def _oa_institution(inst: dict) -> Optional[dict]:
    name = inst.get("display_name")
    key = ror_key(inst.get("ror")) or openalex_key(inst.get("id")) or institution_name_key(name)
    if not key:
        return None
    return {"key": key, "name": name, "country": country_code(inst.get("country_code"))}


def actors_from_openalex(authorships: Optional[Sequence[dict]]) -> Optional[str]:
    """Actors of an OpenAlex work from its ``authorships`` list.

    Authors are keyed by ORCID, else OpenAlex author id; institutions by ROR,
    else OpenAlex institution id. An authorship OpenAlex matched to no
    institution falls back on its raw affiliation strings.
    """
    authors, institutions, countries = [], [], []
    for a in authorships or []:
        person = a.get("author") or {}
        name = person.get("display_name") or a.get("raw_author_name")
        key = (orcid_key(person.get("orcid")) or openalex_key(person.get("id"))
               or author_name_key(name))
        if key:
            authors.append({"key": key, "name": name})
        insts = [i for i in (_oa_institution(x) for x in a.get("institutions") or []) if i]
        if not insts:
            for raw in a.get("raw_affiliation_strings") or []:
                name_ = institution_from_address(raw)
                if name_:
                    insts.append({"key": institution_name_key(name_), "name": name_,
                                  "country": country_from_address(raw)})
        institutions.extend(insts)
        countries.extend(country_code(c) for c in a.get("countries") or [])
    return make_actors(authors, institutions, countries)


# ---- OpenAlex (CSV export) ---------------------------------------------------

def _pipe_split(value) -> List[str]:
    return value.split("|") if isinstance(value, str) else []


_OA_CSV_DICT_SPLIT = re.compile(r"\|(?=\{)")


def _oa_csv_institutions(value) -> List[dict]:
    """Parse the flattened ``authorships.institutions`` cell: python-repr
    dicts joined by ``|``, with empty tokens for unmatched authorships."""
    out = []
    if not isinstance(value, str):
        return out
    for token in _OA_CSV_DICT_SPLIT.split(value):
        token = token.strip("|").strip()
        if not token.startswith("{"):
            continue
        try:
            inst = ast.literal_eval(token)
        except (ValueError, SyntaxError):
            continue
        parsed = _oa_institution(inst) if isinstance(inst, dict) else None
        if parsed:
            out.append(parsed)
    return out


def actors_from_openalex_csv(row) -> Optional[str]:
    """Actors of one row of an OpenAlex CSV export.

    The export flattens each authorship field with ``|``: author ids, names and
    ORCIDs stay aligned per author (ORCIDs are used only when their count
    matches), whereas institutions and countries are concatenated across
    authors — read here as the paper-level sets they still reliably are.
    """
    ids = _pipe_split(row.get("authorships.author.id"))
    names = _pipe_split(row.get("authorships.author.display_name"))
    orcids = _pipe_split(row.get("authorships.author.orcid"))
    if len(orcids) != len(ids):
        orcids = [""] * len(ids)
    if len(names) != len(ids):
        names = (names + [""] * len(ids))[:len(ids)] if ids else names
    authors = []
    for i, name in enumerate(names):
        key = (orcid_key(orcids[i]) if i < len(orcids) else None) \
            or (openalex_key(ids[i]) if i < len(ids) else None) \
            or author_name_key(name)
        if key:
            authors.append({"key": key, "name": name or None})
    countries = [country_code(c) for c in _pipe_split(row.get("authorships.countries"))]
    return make_actors(authors, _oa_csv_institutions(row.get("authorships.institutions")),
                       countries)


#: The OpenAlex CSV columns :func:`actors_from_openalex_csv` reads.
OA_CSV_ACTOR_COLUMNS = ("authorships.author.id", "authorships.author.display_name",
                        "authorships.author.orcid", "authorships.institutions",
                        "authorships.countries")


# ---- Web of Science (Expanded API record) ------------------------------------

def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _wos_person_orcid(name: dict) -> Optional[str]:
    key = orcid_key(name.get("orcid_id"))
    if key:
        return key
    items = _as_list((name.get("data-item-ids") or {}).get("data-item-id"))
    for wanted in ("PreferredORCID", "OtherORCID"):
        for it in items:
            if isinstance(it, dict) and it.get("id-type") == wanted:
                key = orcid_key(it.get("content"))
                if key:
                    return key
    return None


def _wos_organization(spec: dict) -> Optional[dict]:
    """The unified organisation of a WoS ``address_spec``: the preferred
    (``pref: "Y"``) top-level organisation with its ROR, else the raw one."""
    orgs = [o for o in _as_list((spec.get("organizations") or {}).get("organization"))
            if isinstance(o, dict) and o.get("content")]
    preferred = [o for o in orgs if o.get("pref") == "Y"]
    preferred.sort(key=lambda o: o.get("level", 0) or 0)
    org = preferred[0] if preferred else (orgs[0] if orgs else None)
    name = org.get("content") if org else institution_from_address(spec.get("full_address"))
    if not name:
        return None
    name = str(name)
    key = (ror_key(org.get("ror_id")) if org else None) or institution_name_key(name)
    country = country_code(spec.get("country")) or country_from_address(spec.get("full_address"))
    return {"key": key, "name": name, "country": country}


def actors_from_wos_record(record: dict) -> Optional[str]:
    """Actors of a WoS Expanded API record.

    Authors (``role: author``) are keyed by ORCID, else WoS person id
    (``daisng_id``); institutions are the unified organisation of each address,
    keyed by its ROR when WoS gives one.
    """
    static = record.get("static_data") or {}
    names = _as_list(((static.get("summary") or {}).get("names") or {}).get("name"))
    authors = []
    for n in names:
        if not isinstance(n, dict) or n.get("role", "author") != "author":
            continue
        display = n.get("display_name") or n.get("full_name") or n.get("wos_standard")
        dais = n.get("daisng_id")
        key = (_wos_person_orcid(n) or (f"wos:{dais}" if dais else None)
               or author_name_key(display))
        if key:
            authors.append({"key": key, "name": display})
    addresses = _as_list(((static.get("fullrecord_metadata") or {})
                          .get("addresses") or {}).get("address_name"))
    institutions = []
    for a in addresses:
        spec = a.get("address_spec") if isinstance(a, dict) else None
        if isinstance(spec, dict):
            inst = _wos_organization(spec)
            if inst:
                institutions.append(inst)
    return make_actors(authors, institutions)


# ---- Web of Science (BibTeX export) ------------------------------------------

_WOS_BLOCK_SPLIT = re.compile(r"\.\s{2,}|\.\s*$|\n")


def wos_address_blocks(raw) -> List[tuple]:
    """Parse a raw WoS ``Affiliation`` field into ``(names, address)`` blocks.

    Each block reads ``"Last, First; Last, First, Org, Suborg, City, Country."``
    (C1 convention: the organisation is the first address segment). Reprint
    blocks (``"(Corresponding Author)"``) repeat a full block with an
    abbreviated name and are kept only when the record has nothing else.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    blocks, reprints = [], []
    for block in _WOS_BLOCK_SPLIT.split(raw.replace("\\&", "&")):
        block = block.strip(" ;")
        if not block:
            continue
        target = reprints if _CORRESPONDING_RE.search(block) else blocks
        block = _CORRESPONDING_RE.sub("", block)
        parts = [p.strip() for p in block.split(";")]
        head, last = parts[:-1], parts[-1]
        pieces = [p.strip() for p in last.split(",")]
        if len(pieces) < 3:                      # no "Last, First, address"
            target.append(([], block))
            continue
        names = head + [f"{pieces[0]}, {pieces[1]}"]
        target.append((names, ", ".join(pieces[2:])))
    return blocks or reprints


def _wos_orcid_map(value) -> Dict[str, str]:
    """``{person_name_norm: orcid_key}`` from a WoS ``ORCID-Numbers`` field
    (``"han, soonhung/0000-0001-5676-8121; ..."``)."""
    out = {}
    if not isinstance(value, str):
        return out
    for item in value.split(";"):
        if "/" not in item:
            continue
        name, oid = item.rsplit("/", 1)
        norm, key = person_name_norm(name, initials=False), orcid_key(oid)
        if norm and key:
            out[norm] = key
    return out


def actors_from_wos_file(author, affiliation_raw, organizations=None,
                         orcids=None) -> Optional[str]:
    """Actors of a WoS BibTeX record.

    *author* is the ``' and '``-separated author field, *affiliation_raw* the
    raw ``Affiliation`` field (address blocks), *organizations* the
    ``'; '``-separated organisation-enhanced ``Affiliations`` field and
    *orcids* the ``ORCID-Numbers`` field. Institutions are the
    organisation-enhanced names when present (they are WoS-unified), else each
    block's first address segment; countries come from the address blocks.
    """
    orcid_by_name = _wos_orcid_map(orcids)
    authors = []
    for name in split_authors(author, sep=" and "):
        key = (orcid_by_name.get(person_name_norm(name, initials=False))
               or author_name_key(name))
        authors.append({"key": key, "name": name})

    blocks = wos_address_blocks(affiliation_raw)
    countries = [country_from_address(addr) for _, addr in blocks]
    distinct = {c for c in countries if c}
    single_country = next(iter(distinct)) if len(distinct) == 1 else None

    institutions = []
    orgs = split_addresses(organizations.replace("\\&", "&")) if isinstance(organizations, str) else []
    if orgs:
        for org in orgs:
            institutions.append({"key": institution_name_key(org), "name": org,
                                 "country": single_country})
    else:
        for _, addr in blocks:
            segments = _address_segments(addr)
            if segments:
                institutions.append({"key": institution_name_key(segments[0]),
                                     "name": segments[0],
                                     "country": country_from_address(addr)})
    return make_actors(authors, institutions, countries)
