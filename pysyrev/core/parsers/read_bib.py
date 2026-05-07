"""
read_bib.py — lecture de fichiers bibliographiques Scopus / WoS / PubMed.

Version autonome, sans classe, refondue depuis pbx_custom.py (pyBibX).

API publique :
    data, entries, log = read_bib('mon_fichier.bib', db='scopus', del_duplicated=True)

    data     : pd.DataFrame (colonnes triées alphabétiquement, valeurs manquantes
               laissées en NaN).
    entries  : liste des colonnes avant le remplissage sanity-check.
    log      : liste de chaînes décrivant ce qui a été trouvé (nb de docs, types,
               doublons...).

Dépendances : numpy, pandas, lib standard uniquement.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


# =============================================================================
# Constantes
# =============================================================================

# Traduction des codes de langue PubMed (champ LA) vers le nom anglais.
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

# Colonnes attendues par le pipeline ; les absentes sont créées avec NaN.
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

# Renommage des colonnes brutes Scopus CSV -> schéma interne.
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

# Renommage des clés BibTeX WoS -> schéma interne.
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

# Renommage des clés PubMed -> schéma interne.
# Les clés 'dp' (tronquée 4 chars) et 'la' (traduite) sont traitées à part.
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

# Normalisation des types de documents vers le vocabulaire Scopus (50 appels
# .replace() successifs dans l'original -> un seul dict ici).
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

# Préfixes PubMed (6 caractères) où plusieurs occurrences successives d'un même
# champ doivent être agrégées sur une seule ligne. (préfixe -> séparateur)
PUBMED_MULTIVALUE_PREFIXES = {
    'FAU - ': '; ',
    'AU  - ': ' and ',
    'AUID- ': '; ',
    'AD  - ': '',
    'PT  - ': None,   # PT : pas d'agrégation, juste marqué consommé
}


# =============================================================================
# Helpers — nettoyage de texte
# =============================================================================

def _clean_titles(titles):
    """
    Nettoie une liste de titres pour la détection de doublons.

    Équivalent à l'appel original :
        clear_text(titles, stop_words=[], lowercase=True, rmv_accents=True,
                   rmv_special_chars=True, rmv_numbers=True, rmv_custom_words=[])

    Beaucoup plus court que l'original, car on n'a pas besoin du chargement
    des fichiers de stopwords (appelé ici avec stop_words=[]).
    """
    out = []
    for raw in titles:
        t = str(raw).lower().replace('’', "'")                 # lowercase
        t = re.sub(r"[^a-zA-Z0-9']+", ' ', t)                       # special chars -> espace
        t = unicodedata.normalize('NFD', t).encode('ascii', 'ignore').decode('utf-8')  # accents
        t = re.sub(r'[0-9]', ' ', t)                                # chiffres
        t = ' '.join(t.split())                                     # espaces multiples
        out.append(t)
    return out


# =============================================================================
# Helpers — closures originales sorties au niveau module.
# =============================================================================

def _assign_authors_to_affiliations(authors_str, affiliations_str):
    """Associe chaque auteur à son affiliation par index (PubMed).
    Si affiliations_str est NaN on propage NaN ; si authors_str est NaN on
    renvoie les affiliations telles quelles."""
    if not isinstance(affiliations_str, str):
        return affiliations_str  # propage NaN
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
    """Associe auteurs et affiliations Scopus, en isolant le corresponding author.
    Propage NaN si l'affiliation est absente."""
    authors_str = row['author']
    affiliations_str = row['affiliation']
    corr_addr = row['correspondence_address1']

    if not isinstance(affiliations_str, str):
        return affiliations_str  # propage NaN

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
# Helpers — parsing selon la source.
# =============================================================================

def _read_scopus_csv(path):
    """Lit un export CSV Scopus. Renvoie (data, doc_count)."""
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
            # dtype=object explicite : sinon une colonne 100% NaN devient float64
            # et refuse .str.replace plus loin.
            data[col] = pd.Series(np.nan, index=data.index, dtype=object)

    data = data.reindex(sorted(data.columns), axis=1)
    data['author'] = data['author'].apply(
        lambda x: x.replace(';', ' and ') if isinstance(x, str) else x
    )
    return data, data.shape[0]


def _preprocess_wos_lines(f_list):
    """Fusionne les continuations de ligne WoS (préfixe '   ')."""
    merged = []
    for line in f_list:
        if line[:3] != '   ':
            merged.append(line)
            continue
        if merged[-1].find('Cited-References') == -1:
            merged[-1] = merged[-1] + line
        else:
            # Dans les Cited-References on remplace les ';' par ',' pour
            # préserver le ';' comme séparateur de références.
            merged[-1] = merged[-1] + ';' + line.replace(';', ',')
    return merged


def _preprocess_pubmed_lines(f_list):
    """
    Agrège les lignes multi-champs PubMed. Pour les préfixes de
    PUBMED_MULTIVALUE_PREFIXES, on scanne en avant les occurrences successives
    et on les concatène avec leur séparateur respectif ; les lignes consommées
    sont marquées en passant leur préfixe en minuscules.
    """
    out = []
    for i, line in enumerate(f_list):
        prefix = line[:6]
        prev_prefix = f_list[i - 1][:6] if i > 0 else ''
        low = prefix.lower()
        is_multi = low in ('fau - ', 'au  - ', 'auid- ', 'ad  - ')
        is_pt = low == 'pt  - '

        # Continuation indentée.
        if prefix == '      ':
            out[-1] += line[6:]
            continue

        if i == 0 or (prefix != prev_prefix and not is_multi):
            out.append(line)
            continue

        if prefix == prev_prefix and not is_multi and not is_pt:
            out[-1] += '; ' + line[6:]
            continue

        # Préfixe multi-valeurs : agrégation en avant.
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
                    # Marquage : passer le préfixe en minuscules.
                    f_list[j] = f_list[j][:6].lower() + f_list[j][6:]

    # Normalisation des délimiteurs '-' entre préfixe et valeur en '='.
    for i, line in enumerate(out):
        if len(line) > 4 and line[4] == '-':
            out[i] = line[:4] + '=' + line[5:]
        if out[i][:3] == 'LID':
            out[i] = out[i].replace(' [doi]', '')
    return out


def _parse_bibtex_like(f_list, db):
    """
    Parse un flux BibTeX-like (Scopus .bib, WoS .bib prétraité, PubMed .nbib
    prétraité). Renvoie (lhs, rhs, doc_count). Marqueur 'doc_start' séparant
    les enregistrements.
    """
    lhs, rhs = [], []
    doc = 0
    for line in f_list:
        # Début d'enregistrement : '@xxx' (BibTeX) ou 'PMID' (PubMed).
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
    """Renomme les clés parsées vers le schéma interne. Modifie lhs/rhs en place."""
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
    """
    Construit le DataFrame à partir des listes clés/valeurs.

    ANCIEN : pd.DataFrame vide + data.iloc[count, col] = rhs[i] dans une boucle.
             Très lent (O(N) appels iloc avec overhead pandas).
    NOUVEAU : dict-of-lists rempli en Python pur, puis une seule construction
              de DataFrame. Typiquement 50× à 200× plus rapide sur gros fichiers.
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

    # dtype=object : indispensable pour que les colonnes 100% NaN restent
    # compatibles avec l'accesseur .str plus loin.
    return pd.DataFrame(columns, dtype=object)


# =============================================================================
# Helpers — post-traitements.
# =============================================================================

def _deduplicate(data, doc, log):
    """
    Doublon = DOI dupliqué (hors NaN) OU titre nettoyé dupliqué.
    Version vectorisée par masques booléens.
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
    """Ajoute au log le décompte par type de document (Counter, 1 passe).
    Les NaN sont comptés sous le libellé 'UNKNOWN' uniquement dans le log,
    pas dans les données."""
    types = data['document_type'].fillna('UNKNOWN').tolist()
    counts = Counter(types)
    log.append('')
    for tp in sorted(counts):
        log.append(f'{tp} = {counts[tp]}')


def _fix_wos_affiliations(data):
    """Nettoyage des affiliations WoS (préserve les points derrière une majuscule)."""
    s = data['affiliation_'].str.replace(r'(?<=[A-Z])\.', '#', regex=True)
    s = s.str.replace(';', ',', regex=False)
    s = s.str.replace('.', ';', regex=False)
    s = s.str.rstrip(';')
    s = s.str.replace('#', '.', regex=False)
    data['affiliation_'] = s
    return data


def _rebuild_affiliation_from_affiliation_(data):
    """Reconstitue 'affiliation' à partir de 'affiliation_' (cas WoS dégradé)."""
    mask = data['affiliation'].isna() & data['affiliation_'].notna()
    for i in data.index[mask]:
        s = data.loc[i, 'affiliation_']
        parts = s.split('.')
        parts[0] = re.sub(r'.*?\(Corresponding Author\), ', '', parts[0])
        new_parts = [p.split(',', 1)[-1].strip() for p in parts[1:] if ',' in p]
        new_parts.insert(0, parts[0])
        data.loc[i, 'affiliation'] = '. '.join(new_parts)
    return data


# =============================================================================
# API publique.
# =============================================================================

def read_bib(path, db='scopus', del_duplicated=True) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Lit un fichier bibliographique Scopus / WoS / PubMed.

    Parameters
    ----------
    path : str
        Chemin vers le fichier (.csv, .bib, .nbib).
    db : {'scopus', 'wos', 'pubmed'}
        Base de données source.
    del_duplicated : bool, default True
        Si True, retire les doublons (DOI identique ou titre nettoyé identique).

    Returns
    -------
    data : pd.DataFrame
        Colonnes triées alphabétiquement, valeurs manquantes en NaN.
    entries : list[str]
        Colonnes présentes avant le remplissage sanity-check.
    log : list[str]
        Messages descriptifs (comptes de docs, de types, de doublons).
    """
    db = db.lower()
    log = []
    file_ext = os.path.splitext(path)[1].lower()

    # --- 1. Lecture & parsing selon la source. -------------------------------
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

    # --- 2. Normalisation des types de documents. ----------------------------
    data['document_type'] = data['document_type'].replace(DOCUMENT_TYPE_MAP)

    # --- 3. Déduplication. ---------------------------------------------------
    if del_duplicated and 'doi' in entries:
        data = _deduplicate(data, doc, log)
    else:
        log.append(f'A Total of {doc} Documents were Found')

    # --- 4. WoS : 'type' prime sur 'document_type' si présent. ---------------
    if db == 'wos' and 'type' in entries:
        data['document_type'] = data['type']

    if 'document_type' in entries:
        _report_document_types(data, log)

    # --- 5. Post-traitements communs. ----------------------------------------
    # .str.replace propage NaN, contrairement à .apply(lambda x: x.replace(...))
    # qui crasherait sur NaN.
    data['keywords'] = data['keywords'].str.replace(',', ';', regex=False)
    data['author_keywords'] = data['author_keywords'].str.replace(',', ';', regex=False)

    if db == 'wos':
        if 'affiliation_' not in data.columns:
            data['affiliation_'] = pd.Series(np.nan, index=data.index, dtype=object)
        # Year manquante -> 4 premiers caractères de 'da'.
        mask = data['year'].isna()
        if mask.any() and 'da' in data.columns:
            data.loc[mask, 'year'] = data.loc[mask, 'da'].str[:4]

    if 'affiliation' in data.columns and 'affiliation_' in data.columns:
        data = _rebuild_affiliation_from_affiliation_(data)

    if 'affiliation' in data.columns and 'affiliations' in data.columns:
        mask = data['affiliation'].isna() & data['affiliations'].notna()
        data.loc[mask, 'affiliation'] = data.loc[mask, 'affiliations']

    if db == 'scopus':
        # 'Corresponding Author ' + NaN -> NaN (pandas propage).
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

    # 'UNKN'/'unkn' provenant des exports sont considérés comme manquants.
    data = data.replace(['UNKN', 'unkn'], np.nan)

    for col in ('year', 'cited_by'):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data, entries, log


# =============================================================================
# Command line usage : python read_bib.py <fichier> [db]
# =============================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python read_bib.py <fichier> [scopus|wos|pubmed]')
        sys.exit(1)
    path_ = sys.argv[1]
    db_ = sys.argv[2] if len(sys.argv) > 2 else 'scopus'
    data_, entries_, log_ = read_bib(path_, db=db_)
    for line in log_:
        print(line)
    print(f'\nDataFrame shape: {data_.shape}')
    print(f'Colonnes : {list(data_.columns)}')
