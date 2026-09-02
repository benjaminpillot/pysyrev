"""corpus-map helpers. See SKILL.md for the workflow these support."""
import re

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

    CRITICAL: OpenAlex leaves `language` empty on many in-scope records
    (esp. conference papers). Never exclude on empty language — only an
    explicit non-'en' code excludes when require_english is True.
    """
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
        if lang and lang != "en":   # empty passes; wrong-language excludes
            return False
    return True

def has_screenable_abstract(text, min_chars=80):
    """True if there is enough abstract to LLM-screen on. Below this,
    screen on title/venue/topics or complete from WoS first."""
    return bool(text) and len(text) > min_chars

def salton_coupling(refsets):
    """Salton-normalised bibliographic-coupling matrix from a list of
    reference-id sets. W[i,j] = shared / sqrt(|refs_i| * |refs_j|)."""
    import numpy as np
    n = len(refsets)
    W = np.zeros((n, n))
    for i in range(n):
        a = refsets[i]
        if not a:
            continue
        for j in range(i + 1, n):
            b = refsets[j]
            if not b:
                continue
            shared = len(a & b)
            if shared:
                W[i, j] = W[j, i] = shared / np.sqrt(len(a) * len(b))
    return W

def leiden_clusters(W, resolution=0.5, seed=7, min_size=5):
    """Leiden communities from a coupling matrix. Returns (labels, graph,
    modularity). Nodes below min_size are relabelled -1 (uncoupled tail)."""
    import numpy as np, igraph as ig, leidenalg as la
    n = W.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = [W[i, j] for i, j in edges]
    part = la.find_partition(g, la.RBConfigurationVertexPartition,
                             weights=g.es["weight"],
                             resolution_parameter=resolution, seed=seed)
    labels = np.array(part.membership)
    sizes = {c: int((labels == c).sum()) for c in set(labels)}
    labels = np.array([c if sizes[c] >= min_size else -1 for c in labels])
    return labels, g, part.modularity

def cluster_terms(texts, labels, top=12, ngram=(1, 2), min_df=3, max_df=0.4):
    """Distinguishing TF-IDF terms per cluster (mean tf-idf ranking)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram,
                          min_df=min_df, max_df=max_df)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        m = labels == c
        mean = np.asarray(X[m].mean(axis=0)).ravel()
        out[c] = vocab[np.argsort(-mean)[:top]].tolist()
    return out

def wos_complete(dois, api_key, want_refs=None, timeout=40, pause=0.25):
    """Complete missing abstracts (and optionally cited-reference DOIs) from
    Web of Science Expanded. Returns {doi: {found, abstract, has_abs,
    ref_dois?, ref_count?}}. want_refs: iterable of DOIs to also pull refs for.

    WoS under-indexes preprints and many conference series — use to COMPLETE
    OpenAlex records, never as the primary search.
    """
    import urllib.request, urllib.parse, json, time
    want_refs = set(want_refs or [])
    base = "https://api.clarivate.com/api/wos"
    hdr = {"X-ApiKey": api_key, "Accept": "application/json"}

    def _get(url):
        rq = urllib.request.Request(url, headers=hdr)
        return json.load(urllib.request.urlopen(rq, timeout=timeout))

    def _abstract(rec):
        ab = rec["static_data"].get("fullrecord_metadata", {}).get("abstracts", {})
        if not ab or ab.get("count", 0) == 0:
            return ""
        node = ab["abstract"]
        if isinstance(node, list):
            node = node[0]
        p = node.get("abstract_text", {}).get("p", "")
        if isinstance(p, list):
            p = " ".join(str(x) for x in p)
        return str(p)

    def _refs(uid):
        out, first = [], 1
        while True:
            u = base + "/references?" + urllib.parse.urlencode(
                {"databaseId": "WOS", "uniqueId": uid, "count": 100,
                 "firstRecord": first})
            r = _get(u); D = r.get("Data", [])
            out += D
            tot = r.get("QueryResult", {}).get("RecordsFound", 0)
            if len(out) >= tot or not D:
                break
            first += 100; time.sleep(0.2)
        return out

    result = {}
    for d in dois:
        try:
            u = base + "?" + urllib.parse.urlencode(
                {"databaseId": "WOS", "usrQuery": "DO=(%s)" % d,
                 "count": 1, "firstRecord": 1})
            r = _get(u)
            md = r.get("Data", {}).get("Records", {}).get("records", "")
            recs = md.get("REC", []) if isinstance(md, dict) else []
            if not recs:
                result[d] = {"found": False}
            else:
                rec = recs[0]; ab = _abstract(rec)
                e = {"found": True, "uid": rec["UID"], "abstract": ab,
                     "has_abs": len(ab) > 80}
                if d in want_refs:
                    rf = _refs(rec["UID"])
                    e["ref_dois"] = [x["DOI"].lower() for x in rf if x.get("DOI")]
                    e["ref_count"] = len(rf)
                result[d] = e
        except Exception as ex:
            result[d] = {"found": False, "err": str(ex)[:80]}
        time.sleep(pause)
    return result
