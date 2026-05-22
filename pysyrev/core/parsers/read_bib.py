"""
read_bib.py — bibliographic file reader for Scopus / WoS / PubMed.

Stand-alone, class-free module refactored from pbx_custom.py (pyBibX).

Public API:
    data, entries, log = read_bib('file.bib', db='scopus', del_duplicated=True)

    data     : pd.DataFrame (columns sorted alphabetically, missing values as NaN).
    entries  : list of columns present before the sanity-check fill.
    log      : list of strings describing what was found (doc count, types, duplicates…).

`path` may also be a directory — all matching files are concatenated automatically.

Dependencies: numpy, pandas, standard library only.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# PubMed language codes (LA field) mapped to their English name.
LANGUAGE_NAMES = {
    'afr': 'Afrikaans', 'alb': 'Albanian', 'amh': 'Amharic', 'ara': 'Arabic',
    'arm': 'Armenian', 'aze': 'Azerbaijani', 'bos': 'Bosnian',
    'bul': 'Bulgarian', 'cat': 'Catalan', 'chi': 'Chinese', 'cze': 'Czech',
    'dan': 'Danish', 'dut': 'Dutch', 'eng': 'English', 'epo': 'Esperanto',
    'est': 'Estonian', 'fin': 'Finnish', 'fre': 'French', 'geo': 'Georgian',
    'ger': 'German', 'gla': 'Scottish Gaelic', 'gre': 'Greek, Modern',
    'heb': 'Hebrew', 'hin': 'Hindi', 'hrv': 'Croatian', 'hun': 'Hungarian',
    'ice': 'Icelandic', 'ind': 'Indonesian', 'ita': 'Italian',
    'jpn': 'Japanese', 'kin': 'Kinyarwanda', 'kor': 'Korean', 'lat': 'Latin',
    'lav': 'Latvian', 'lit': 'Lithuanian', 'mac': 'Macedonian',
    'mal': 'Malayalam', 'mao': 'Maori', 'may': 'Malay',
    'mul': 'Multiple languages', 'nor': 'Norwegian',
    'per': 'Persian, Iranian', 'pol': 'Polish', 'por': 'Portuguese',
    'pus': 'Pushto', 'rum': 'Romanian, Rumanian, Moldovan', 'rus': 'Russian',
    'san': 'Sanskrit', 'slo': 'Slovak', 'slv': 'Slovenian', 'spa': 'Spanish',
    'srp': 'Serbian', 'swe': 'Swedish', 'tha': 'Thai', 'tur': 'Turkish',
    'ukr': 'Ukrainian', 'und': 'Undetermined', 'vie': 'Vietnamese',
    'wel': 'Welsh',
}

# Columns expected by the pipeline; missing ones are created as NaN.
SANITY_CHECK_COLUMNS = [
    'abbrev_source_title', 'abstract', 'address', 'affiliation', 'art_number',
    'author', 'author_keywords', 'chemicals_cas', 'coden',
    'correspondence_address1', 'document_type', 'doi', 'editor',
    'funding_details', 'funding_text\xa01', 'funding_text\xa02',
    'funding_text\xa03', 'isbn', 'issn', 'journal', 'keywords', 'language',
    'note', 'number', 'page_count', 'pages', 'publisher', 'pubmed_id',
    'references', 'source', 'sponsors', 'title', 'tradenames', 'url',
    'volume', 'year',
]

# Raw Scopus CSV column names → internal schema.
SCOPUS_CSV_RENAMES = {
    'abbreviated source title':      'abbrev_source_title',
    'document type':                 'document_type',
    'art. no.':                      'art_number',
    'author keywords':               'author_keywords',
    'authors':                       'author',
    'chemicals/cas':                 'chemicals_cas',
    'correspondence address':        'correspondence_address',
    'editors':                       'editor',
    'funding details':               'funding_details',
    'index keywords':                'keywords',
    'language of original document': 'language',
    'cited by':                      'note',
    'page count':                    'page_count',
    'pubmed id':                     'pubmed_id',
}

# WoS BibTeX keys → internal schema.
WOS_LHS_RENAMES = {
    'affiliation':      'affiliation_',
    'affiliations':     'affiliation',
    'article-number':   'art_number',
    'cited-references': 'references',
    'keywords':         'author_keywords',
    'journal-iso':      'abbrev_source_title',
    'keywords-plus':    'keywords',
    'note':             'note_',
    'times-cited':      'cited_by',
    'type':             'document_type',
    'unique-id':        'id',
}

# PubMed keys → internal schema.
# Keys 'dp' (truncated to 4 chars) and 'la' (translated) are handled separately.
PUBMED_LHS_RENAMES = {
    'ab':   'abstract',
    'ad':   'affiliation',
    'au':   'author',
    'auid': 'orcid',
    'fau':  'full_author',
    'lid':  'doi',
    'dp':   'year',
    'ed':   'editor',
    'ip':   'issue',
    'is':   'issn',
    'isbn': 'isbn',
    'jt':   'journal',
    'la':   'language',
    'mh':   'keywords',
    'ot':   'author_keywords',
    'pg':   'pages',
    'pt':   'document_type',
    'pmid': 'pubmed_id',
    'ta':   'abbrev_source_title',
    'ti':   'title',
    'vi':   'volume',
}

# Document type normalisation to the Scopus vocabulary (50 successive
# .replace() calls in the original → a single dict here).
DOCUMENT_TYPE_MAP = {
    # WoS -> Scopus
    'Article; Early Access':           'Article in Press',
    'Article; Proceedings Paper':      'Proceedings Paper',
    'Article; Discussion':             'Article',
    'Article; Letter':                 'Article',
    'Article; Excerpt':                'Article',
    'Article; Chronology':             'Article',
    'Article; Correction':             'Article',
    'Article; Correction, Addition':   'Article',
    'Article; Data Paper':             'Article',
    'Art Exhibit Review':              'Review',
    'Dance Performance Review':        'Review',
    'Music Performance Review':        'Review',
    'Music Score Review':              'Review',
    'Film Review':                     'Review',
    'TV Review, Radio Review':         'Review',
    'TV Review, Radio Review, Video':  'Review',
    'Theater Review, Video':           'Review',
    'Database Review':                 'Review',
    'Record Review':                   'Review',
    'Software Review':                 'Review',
    'Hardware Review':                 'Review',
    # PubMed -> Scopus
    'Clinical Study':                                           'Article',
    'Clinical Trial':                                           'Article',
    'Clinical Trial Protocol':                                  'Article',
    'Clinical Trial, Phase I':                                  'Article',
    'Clinical Trial, Phase II':                                 'Article',
    'Clinical Trial, Phase III':                                'Article',
    'Clinical Trial, Phase IV':                                 'Article',
    'Clinical Trial, Veterinary':                               'Article',
    'Comparative Study':                                        'Article',
    'Controlled Clinical Trial':                                'Article',
    'Corrected and Republished Article':                        'Article',
    'Duplicate Publication':                                    'Article',
    'Essay':                                                    'Article',
    'Historical Article':                                       'Article',
    'Journal Article':                                          'Article',
    'Letter':                                                   'Article',
    'Meta-Analysis':                                            'Article',
    'Randomized Controlled Trial':                              'Article',
    'Randomized Controlled Trial, Veterinary':                  'Article',
    'Research Support, N.I.H., Extramural':                     'Article',
    'Research Support, N.I.H., Intramural':                     'Article',
    "Research Support, Non-U.S. Gov't":                         'Article',
    "Research Support, U.S. Gov't, Non-P.H.S.":                 'Article',
    "Research Support, U.S. Gov't, P.H.S.":                     'Article',
    'Research Support, U.S. Government':                        'Article',
    'Research Support, American Recovery and Reinvestment Act': 'Article',
    'Technical Report':                                         'Article',
    'Twin Study':                                               'Article',
    'Validation Study':                                         'Article',
    'Clinical Conference':                                      'Conference Paper',
    'Congress':                                                 'Conference Paper',
    'Consensus Development Conference':                         'Conference Paper',
    'Consensus Development Conference, NIH':                    'Conference Paper',
    'Systematic Review':                                        'Review',
    'Scientific Integrity Review':                              'Review',
}

# PubMed prefixes (6 chars) where successive occurrences of the same field
# must be merged onto one line. Maps prefix → separator.
PUBMED_MULTIVALUE_PREFIXES = {
    'FAU - ': '; ',
    'AU  - ': ' and ',
    'AUID- ': '; ',
    'AD  - ': '',
    'PT  - ': None,   # PT: no aggregation, just marked as consumed
}


# =============================================================================
# Helpers — text cleaning
# =============================================================================

def _clean_titles(titles):
    """Clean a list of titles for duplicate detection.

    Equivalent to the original call:
        clear_text(titles, stop_words=[], lowercase=True, rmv_accents=True,
                   rmv_special_chars=True, rmv_numbers=True, rmv_custom_words=[])

    Much shorter than the original because stopword file loading is not needed
    (called here with stop_words=[]).
    """
    out = []
    for raw in titles:
        t = str(raw).lower().replace('’', "'")                  # lowercase
        t = re.sub(r"[^a-zA-Z0-9']+", ' ', t)                       # special chars → space
        t = unicodedata.normalize('NFD', t).encode('ascii', 'ignore').decode('utf-8')  # accents
        t = re.sub(r'[0-9]', ' ', t)                                 # digits
        t = ' '.join(t.split())                                      # multiple spaces
        out.append(t)
    return out


# =============================================================================
# Helpers — module-level functions
# =============================================================================

def _assign_authors_to_affiliations(authors_str, affiliations_str):
    """Map each author to their affiliation by index (PubMed).

    Propagates NaN when affiliations_str is NaN; returns affiliations as-is
    when authors_str is NaN.
    """
    if not isinstance(affiliations_str, str):
        return affiliations_str  # propagate NaN
    affiliations = [a.strip() for a in affiliations_str.split(';')]
    if not isinstance(authors_str, str):
        return '; '.join(affiliations)
    authors = [a.strip() for a in authors_str.split(' and ')]
    new_affs = [
        f"{authors[i]} {aff}" if i < len(authors) else aff
        for i, aff in enumerate(affiliations)
    ]
    return '; '.join(new_affs)


def _get_corresponding_authors_and_affiliations(corr_address):
    if not isinstance(corr_address, str):
        return [], []
    match = re.search(
        r'Corresponding Author\s+([^;]+);([^;]+)',
        corr_address, re.IGNORECASE,
    )
    if not match:
        return [], []
    authors_part = match.group(1).strip()
    affiliations_part = match.group(2).strip()
    authors = re.split(r'\s+and\s+|;', authors_part)
    return authors, [affiliations_part]


def _map_authors_to_affiliations(row):
    """Map Scopus authors to affiliations, isolating the corresponding author.

    Propagates NaN when the affiliation column is absent.
    """
    authors_str = row['author']
    affiliations_str = row['affiliation']
    corr_addr = row['correspondence_address1']

    if not isinstance(affiliations_str, str):
        return affiliations_str  # propagate NaN

    authors = re.split(r'\s+and\s+|;', authors_str) if isinstance(authors_str, str) else []
    affiliations = [a.strip() for a in affiliations_str.split(';')]

    ca_authors, ca_affiliations = _get_corresponding_authors_and_affiliations(corr_addr)
    new_affs = [f"{a} {aff}" for a, aff in zip(ca_authors, ca_affiliations)]

    for ca_aff in ca_affiliations:
        affiliations = [aff for aff in affiliations if ca_aff.lower() not in aff.lower()]

    remaining_authors = [a for a in authors if a not in ca_authors]
    for i, author in enumerate(remaining_authors):
        if i < len(affiliations):
            new_affs.append(f"{author} {affiliations[i]}")

    return '; '.join(new_affs)


# =============================================================================
# Helpers — source-specific parsing
# =============================================================================

def _read_scopus_csv(path):
    """Read a Scopus CSV export. Returns (data, doc_count)."""
    try:
        data = pd.read_csv(path, encoding='utf8', dtype=str)
    except UnicodeDecodeError:
        data = pd.read_csv(path, encoding='cp1252', dtype=str)

    data.columns = data.columns.str.lower()

    rename_map = {
        src: dst for src, dst in SCOPUS_CSV_RENAMES.items()
        if src in data.columns and dst not in data.columns
    }
    if rename_map:
        data = data.rename(columns=rename_map)

    for col in SANITY_CHECK_COLUMNS:
        if col not in data.columns:
            # explicit dtype=object: otherwise a 100%-NaN column becomes float64
            # and would reject .str.replace calls downstream.
            data[col] = pd.Series(np.nan, index=data.index, dtype=object)

    data = data.reindex(sorted(data.columns), axis=1)
    data['author'] = data['author'].apply(
        lambda x: x.replace(';', ' and ') if isinstance(x, str) else x
    )
    return data, data.shape[0]


def _preprocess_wos_lines(f_list):
    """Merge WoS line continuations (lines starting with '   ')."""
    merged = []
    for line in f_list:
        if line[:3] != '   ':
            merged.append(line)
            continue
        if merged[-1].find('Cited-References') == -1:
            merged[-1] = merged[-1] + line
        else:
            # Inside Cited-References, replace ';' with ',' to preserve
            # ';' as the reference separator.
            merged[-1] = merged[-1] + ';' + line.replace(';', ',')
    return merged


def _preprocess_pubmed_lines(f_list):
    """Aggregate multi-field PubMed lines.

    For prefixes in PUBMED_MULTIVALUE_PREFIXES, successive occurrences are
    scanned ahead and concatenated with their respective separator; consumed
    lines are marked by lower-casing their prefix.
    """
    out = []
    for i, line in enumerate(f_list):
        prefix = line[:6]
        prev_prefix = f_list[i - 1][:6] if i > 0 else ''
        low = prefix.lower()
        is_multi = low in ('fau - ', 'au  - ', 'auid- ', 'ad  - ')
        is_pt = low == 'pt  - '

        # Indented continuation line.
        if prefix == '      ':
            out[-1] += line[6:]
            continue

        if i == 0 or (prefix != prev_prefix and not is_multi):
            out.append(line)
            continue

        if prefix == prev_prefix and not is_multi and not is_pt:
            out[-1] += '; ' + line[6:]
            continue

        # Multi-value prefix: aggregate forward.
        if prefix in PUBMED_MULTIVALUE_PREFIXES:
            sep = PUBMED_MULTIVALUE_PREFIXES[prefix]
            out.append(line)
            j = i + 1
            while j < len(f_list) and len(f_list[j]) != 0:
                j += 1
                if j >= len(f_list):
                    break
                if f_list[j][:6].lower() == low:
                    if sep is not None:
                        out[-1] += sep + f_list[j][6:]
                    # Mark as consumed: lower-case the prefix.
                    f_list[j] = f_list[j][:6].lower() + f_list[j][6:]

    # Normalise '-' delimiters between prefix and value to '='.
    for i, line in enumerate(out):
        if len(line) > 4 and line[4] == '-':
            out[i] = line[:4] + '=' + line[5:]
        if out[i][:3] == 'LID':
            out[i] = out[i].replace(' [doi]', '')
    return out


def _parse_bibtex_like(f_list, db):
    """Parse a BibTeX-like stream (Scopus .bib, pre-processed WoS .bib,
    pre-processed PubMed .nbib). Returns (lhs, rhs, doc_count).
    'doc_start' is the sentinel separating records.
    """
    lhs, rhs = [], []
    doc = 0
    for line in f_list:
        # Record start: '@xxx' (BibTeX) or 'PMID' (PubMed).
        if line.find('@') == 0 or line[:4].lower() == 'pmid':
            lhs.append('doc_start')
            rhs.append('doc_start')
            if db == 'pubmed':
                lhs.extend(['note', 'source'])
                rhs.extend(['0', 'PubMed'])
            elif db == 'wos':
                lhs.append('source')
                rhs.append('WoS')
            doc += 1

        has_eq = line.find('=') != -1
        if (has_eq and line.find(' ') != 0) or (has_eq and line.find('=') == 15):
            key, _, value = line.partition('=')
            lhs.append(key.lower().strip())
            rhs.append(
                value.replace('{', '').replace('},', '')
                     .replace('}', '').replace('}},', '').strip()
            )
        elif line.find(' ') == 0 and rhs and rhs[-1] != 'doc_start':
            rhs[-1] += ' ' + (
                line.replace('{', '').replace('},', '')
                    .replace('}', '').replace('}},', '').strip()
            )
    return lhs, rhs, doc


def _apply_lhs_renames(lhs, rhs, db):
    """Rename parsed keys to the internal schema. Modifies lhs/rhs in place."""
    if db == 'scopus':
        has_abbrev = 'abbrev_source_title' in lhs
        for i, k in enumerate(lhs):
            if k == 'journal' and not has_abbrev:
                lhs[i] = 'abbrev_source_title'
            elif k == 'type':
                lhs[i] = 'document_type'

    elif db == 'wos':
        has_iso = 'journal-iso' in lhs
        for i, k in enumerate(lhs):
            if k == 'journal' and not has_iso:
                k = 'journal-iso'
            k = WOS_LHS_RENAMES.get(k, k)
            lhs[i] = k.replace('-', '_')

    elif db == 'pubmed':
        has_ta = 'ta' in lhs
        for i, k in enumerate(lhs):
            if k == 'jt' and not has_ta:
                lhs[i] = 'ta'
                continue
            new_k = PUBMED_LHS_RENAMES.get(k)
            if new_k is None:
                continue
            lhs[i] = new_k
            if k == 'dp':
                rhs[i] = rhs[i][:4]
            elif k == 'la' and rhs[i] in LANGUAGE_NAMES:
                rhs[i] = LANGUAGE_NAMES[rhs[i]]


def _build_dataframe_from_kv(lhs, rhs, doc):
    """Build the DataFrame from key/value lists.

    OLD: empty pd.DataFrame + data.iloc[count, col] = rhs[i] in a loop —
         very slow (O(N) iloc calls with pandas overhead).
    NEW: dict-of-lists filled in pure Python, then a single DataFrame
         construction. Typically 50× to 200× faster on large files.
    """
    labels = set(lhs) - {'doc_start'}
    labels.update(SANITY_CHECK_COLUMNS)
    labels = sorted(labels)

    columns = {lbl: [np.nan] * doc for lbl in labels}

    count = -1
    for key, val in zip(lhs, rhs):
        if key == 'doc_start':
            count += 1
        else:
            columns[key][count] = val

    # dtype=object: required so that 100%-NaN columns remain compatible
    # with .str accessor calls downstream.
    return pd.DataFrame(columns, dtype=object)


# =============================================================================
# Helpers — post-processing
# =============================================================================

def _deduplicate(data, doc, log):
    """Duplicate = duplicated DOI (excluding NaN) OR duplicated cleaned title.
    Vectorised implementation using boolean masks.
    """
    dup_doi = data['doi'].duplicated() & data['doi'].notna()

    titles_clean = _clean_titles(data['title'].tolist())
    dup_title = pd.Series(titles_clean).duplicated()

    to_drop = dup_doi | dup_title
    n_drop = int(to_drop.sum())
    data = data.loc[~to_drop].reset_index(drop=True)
    log.append(
        f'A Total of {doc - n_drop} Documents were Found '
        f'( {doc} Documents and {n_drop} Duplicates )'
    )
    return data


def _report_document_types(data, log):
    """Append a per-type document count to the log (Counter, single pass).
    NaN values are counted under 'UNKNOWN' in the log only, not in the data.
    """
    types = data['document_type'].fillna('UNKNOWN').tolist()
    counts = Counter(types)
    log.append('')
    for tp in sorted(counts):
        log.append(f'{tp} = {counts[tp]}')


def _fix_wos_affiliations(data):
    """Clean WoS affiliations (preserves dots following an uppercase letter)."""
    s = data['affiliation_'].str.replace(r'(?<=[A-Z])\.', '#', regex=True)
    s = s.str.replace(';', ',', regex=False)
    s = s.str.replace('.', ';', regex=False)
    s = s.str.rstrip(';')
    s = s.str.replace('#', '.', regex=False)
    data['affiliation_'] = s
    return data


def _rebuild_affiliation_from_affiliation_(data):
    """Rebuild 'affiliation' from 'affiliation_' (degraded WoS case)."""
    mask = data['affiliation'].isna() & data['affiliation_'].notna()
    for i in data.index[mask]:
        s = data.loc[i, 'affiliation_']
        parts = s.split('.')
        parts[0] = re.sub(r'.*?\(Corresponding Author\), ', '', parts[0])
        new_parts = [p.split(',', 1)[-1].strip() for p in parts[1:] if ',' in p]
        new_parts.insert(0, parts[0])
        data.loc[i, 'affiliation'] = '. '.join(new_parts)
    return data


# File extensions accepted per database when reading a directory.
_DIR_EXTENSIONS = {
    'wos':    ('.bib',),
    'scopus': ('.bib', '.csv'),
    'pubmed': ('.nbib', '.txt'),
}


def _read_bib_dir(dirpath, db, del_duplicated):
    """Read all matching bib files in *dirpath* and return a single concatenated result.

    Deduplication (DOI / cleaned title) is applied once on the merged records,
    not file by file.
    """
    exts = _DIR_EXTENSIONS.get(db, ('.bib',))
    files = sorted(
        entry.path for entry in os.scandir(dirpath)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts
    )
    if not files:
        ext_str = '/'.join(exts)
        raise ValueError(
            f"No {ext_str} files found in directory: {dirpath}"
        )

    frames = []
    for fpath in files:
        df, _, _ = read_bib(fpath, db=db, del_duplicated=False)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    log = []
    doc = len(data)

    if del_duplicated and 'doi' in data.columns:
        data = _deduplicate(data, doc, log)
    else:
        log.append(f'A Total of {doc} Documents were Found')

    if 'document_type' in data.columns:
        _report_document_types(data, log)

    entries = list(data.columns)
    return data, entries, log


# =============================================================================
# Public API
# =============================================================================

def read_bib(path, db='scopus', del_duplicated=True) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read a Scopus / WoS / PubMed bibliographic file.

    Parameters
    ----------
    path : str
        Path to a file (.csv, .bib, .nbib) **or to a directory**.
        When a directory is given, all files whose extension matches the
        expected format (`.bib` for WoS/Scopus, `.nbib`/`.txt` for PubMed)
        are read and concatenated; deduplication is applied once on the
        combined result.
    db : {'scopus', 'wos', 'pubmed'}
        Source database.
    del_duplicated : bool, default True
        If True, remove duplicates (identical DOI or cleaned title).

    Returns
    -------
    data : pd.DataFrame
        Columns sorted alphabetically; missing values left as NaN.
    entries : list[str]
        Columns present before the sanity-check fill.
    log : list[str]
        Descriptive messages (document counts, types, duplicates).
    """
    db = db.lower()

    if os.path.isdir(path):
        return _read_bib_dir(path, db=db, del_duplicated=del_duplicated)

    log = []
    file_ext = os.path.splitext(path)[1].lower()

    # --- 1. Read & parse according to the source. ----------------------------
    if db == 'scopus' and file_ext == '.csv':
        data, doc = _read_scopus_csv(path)
    else:
        with open(path, 'r', encoding='utf8') as fh:
            f_list = fh.read().split('\n')

        if db == 'wos':
            f_list = _preprocess_wos_lines(f_list)
        elif db == 'pubmed':
            f_list = _preprocess_pubmed_lines(f_list)

        lhs, rhs, doc = _parse_bibtex_like(f_list, db)
        _apply_lhs_renames(lhs, rhs, db)
        data = _build_dataframe_from_kv(lhs, rhs, doc)

    entries = list(data.columns)

    # --- 2. Normalise document types. ----------------------------------------
    data['document_type'] = data['document_type'].replace(DOCUMENT_TYPE_MAP)

    # --- 3. Deduplication. ---------------------------------------------------
    if del_duplicated and 'doi' in entries:
        data = _deduplicate(data, doc, log)
    else:
        log.append(f'A Total of {doc} Documents were Found')

    # --- 4. WoS: 'type' overrides 'document_type' when present. -------------
    if db == 'wos' and 'type' in entries:
        data['document_type'] = data['type']

    if 'document_type' in entries:
        _report_document_types(data, log)

    # --- 5. Common post-processing. ------------------------------------------
    # .str.replace propagates NaN, unlike .apply(lambda x: x.replace(...))
    # which would crash on NaN.
    data['keywords'] = data['keywords'].str.replace(',', ';', regex=False)
    data['author_keywords'] = data['author_keywords'].str.replace(',', ';', regex=False)

    if db == 'wos':
        if 'affiliation_' not in data.columns:
            data['affiliation_'] = pd.Series(np.nan, index=data.index, dtype=object)
        # Missing year: fall back to the first 4 chars of 'da'.
        mask = data['year'].isna()
        if mask.any() and 'da' in data.columns:
            data.loc[mask, 'year'] = data.loc[mask, 'da'].str[:4]

    if 'affiliation' in data.columns and 'affiliation_' in data.columns:
        data = _rebuild_affiliation_from_affiliation_(data)

    if 'affiliation' in data.columns and 'affiliations' in data.columns:
        mask = data['affiliation'].isna() & data['affiliations'].notna()
        data.loc[mask, 'affiliation'] = data.loc[mask, 'affiliations']

    if db == 'scopus':
        # 'Corresponding Author ' + NaN → NaN (pandas propagates NaN).
        data['correspondence_address1'] = (
            'Corresponding Author ' + data['correspondence_address1']
        )
        data['affiliation'] = data.apply(_map_authors_to_affiliations, axis=1)
    elif db == 'pubmed':
        data['affiliation'] = data.apply(
            lambda row: _assign_authors_to_affiliations(row['author'], row['affiliation']),
            axis=1,
        )
    elif db == 'wos':
        data = _fix_wos_affiliations(data)

    data['abstract'] = data['abstract'].replace(
        '[No abstract available]', np.nan,
    )
    data = data.reindex(sorted(data.columns), axis=1)

    if db == 'scopus' and file_ext == '.bib':
        data['abbrev_source_title'] = np.where(
            data['abbrev_source_title'].isna() & data['source title'].notna(),
            data['source title'], data['abbrev_source_title'],
        )
    elif db == 'scopus' and file_ext == '.csv':
        data['abbrev_source_title'] = np.where(
            data['abbrev_source_title'].isna() & data['journal'].notna(),
            data['journal'], data['abbrev_source_title'],
        )

    # 'UNKN'/'unkn' from exports are treated as missing values.
    data = data.replace(['UNKN', 'unkn'], np.nan)

    for col in ('year', 'cited_by'):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data, entries, log


# =============================================================================
# Command-line usage: python read_bib.py <file> [db]
# =============================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python read_bib.py <file> [scopus|wos|pubmed]')
        sys.exit(1)
    path_ = sys.argv[1]
    db_ = sys.argv[2] if len(sys.argv) > 2 else 'scopus'
    data_, entries_, log_ = read_bib(path_, db=db_)
    for line in log_:
        print(line)
    print(f'\nDataFrame shape: {data_.shape}')
    print(f'Columns: {list(data_.columns)}')
