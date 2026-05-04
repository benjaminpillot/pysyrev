import tempfile

import numpy as np
import pandas as pd

from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype, is_object_dtype
from rapidfuzz import process as rf_process
from tqdm import tqdm

from pysyrev.core.parsers.read_bib import read_bib


# Column rename map for OpenAlex CSV exports (flat column names).
FROM_OA = {"keywords.display_name": "author_keywords",
           "cited_by_count": "cited_by",
           "publication_year": "year",
           "primary_location.source.display_name": "journal",
           "type": "document_type",
           "referenced_works": "references"}


# Each field is described by a semantic kind rather than a concrete dtype.
#   'string'  : any dtype that stores strings (object, StringDtype, ArrowDtype[str])
#   'numeric' : any numeric dtype (int, float, Int64, Float64, pyarrow int/float)
DEFAULT_FIELDS = {
    'id':              'string',
    'abstract':        'string',
    'journal':         'string',
    'title':           'string',
    'author_keywords': 'string',
    'doi':             'string',
    'language':        'string',
    'document_type':   'string',
    'references':      'string',
    'year':            'numeric',
    'cited_by':        'numeric',
}


def _is_string_like(series):
    """
    Pragmatic 'string-like' check: accept anything that stores text, including
    plain ``object`` dtype with NaN values.

    pandas' own ``is_string_dtype`` returns False on ``object`` Series that
    contain NaN, which would reject any real-world bibliographic column. We
    relax that: if the column is ``object``, ``string``, ``str``, or any
    Arrow-string variant, it qualifies.
    """
    return is_object_dtype(series) or is_string_dtype(series)


_KIND_CHECKS = {
    'string':  _is_string_like,
    'numeric': is_numeric_dtype,
}


def check_bib_dataset(dataset):
    """
    Validate that `dataset` conforms to the expected bibliographic schema.

    Checks:
      1. All required columns are present.
      2. Each column has a dtype compatible with its expected kind.

    Raises
    ------
    ValueError
        If any required column is missing, with the list of missing columns.
        If any column has an incompatible dtype, with the offending (column,
        dtype, expected kind) triples.

    Returns
    -------
    pd.DataFrame
        The input dataset, unchanged, if all checks pass.
    """
    missing = [col for col in DEFAULT_FIELDS if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    bad_dtypes = []
    for col, kind in DEFAULT_FIELDS.items():
        check = _KIND_CHECKS[kind]
        if not check(dataset[col]):
            bad_dtypes.append((col, str(dataset[col].dtype), kind))

    if bad_dtypes:
        details = ', '.join(f"{col}: got {dt}, expected {kind}"
                            for col, dt, kind in bad_dtypes)
        raise ValueError(f"Incompatible dtypes -> {details}")

    return dataset


def extract_documents(dataset, doc_type, language, year, nb_citations,
                      scorer, score_cutoff):
    """
    Extract documents matching one or more document types and (optionally) one
    or more languages, filtered by minimum year and citation count.

    Parameters
    ----------
    dataset : pd.DataFrame
        Must contain `document_type`, `language`, `year`, `cited_by`.
    doc_type : str or iterable of str
        Document type queries (fuzzy-matched against unique values of
        `dataset.document_type`).
    language : str or iterable of str or None
        Language queries (fuzzy-matched against unique values of
        `dataset.language`). If None or empty, no language filter is applied.
    year : float or int
        Minimum publication year (inclusive).
    nb_citations : float or int
        Minimum citation count (inclusive).
    scorer : callable
        rapidfuzz scorer (e.g. fuzz.WRatio, fuzz.token_set_ratio).
    score_cutoff : int or float
        Minimum score to accept a fuzzy match (0-100).

    Returns
    -------
    pd.DataFrame
        Documents matching all filters, with duplicate rows removed.
    """

    def _as_list(x):
        """Accept a single string or an iterable of strings and return a list."""
        if isinstance(x, str):
            return [x]
        return list(x)

    def _resolve_categories(queries, candidates):
        """
        For each user query, return every candidate from `candidates` that scores
        at or above `score_cutoff`. Operates on the unique values of a column.

        Returns a set of canonical category values that the user wants to keep.
        """
        accepted = set()
        if candidates.size == 0:
            return accepted
        candidate_list = candidates.tolist()

        for q in queries:
            matches = rf_process.extract(
                q, candidate_list,
                scorer=scorer,
                score_cutoff=score_cutoff,
                limit=None,
            )
            # `extract` returns (choice, score, index); we only need the choice.
            accepted.update(m[0] for m in matches)

        return accepted

    # Numeric filters first — they're cheap and usually very selective.
    filtered = dataset[(dataset.year >= year) & (dataset.cited_by >= nb_citations)]

    # Document type filter: fuzzy on unique values, then .isin.
    doc_type_queries = _as_list(doc_type)
    unique_types = filtered['document_type'].dropna().unique()
    accepted_types = _resolve_categories(
        doc_type_queries, unique_types
    )
    if not accepted_types:
        return filtered.iloc[0:0]  # empty result with correct schema
    mask = filtered['document_type'].isin(accepted_types)

    # Language filter (optional): same trick.
    if language is not None:
        language_queries = _as_list(language)
        if language_queries:
            unique_langs = filtered['language'].dropna().unique()
            accepted_langs = _resolve_categories(
                language_queries, unique_langs
            )
            if not accepted_langs:
                return filtered.iloc[0:0]
            mask &= filtered['language'].isin(accepted_langs)

    result = filtered[mask]

    # Defensive de-duplication: if the user accidentally passed near-synonyms
    # in doc_type or language, a document could have been selected via two
    # different queries. With .isin() this cannot happen anymore (a row is
    # either in or out), but we keep a drop_duplicates on the full row just
    # in case upstream produced duplicates.
    return result.drop_duplicates(ignore_index=True)



def fetch_citations(doi):
    from crossref.restful import Works
    from semanticscholar import SemanticScholar
    works = Works()
    sch = SemanticScholar()
    citation_count = []

    for doi_ in tqdm(doi, desc="Fetch citations"):
        if doi_.lower() != "unknown":
            try:
                paper = sch.get_paper(doi_)
                citation_count.append(paper.citationCount)
            except:
                try:
                    paper = works.doi(doi_)
                    citation_count.append(paper["is-referenced-by-count"])
                except:
                    citation_count.append(np.nan)
        else:
            citation_count.append(np.nan)

    return citation_count


def generate_bib(bibfile, db, del_duplicated, print_log):

    try:
        _, _bibfile = tempfile.mkstemp(suffix=".csv", dir=tempfile.tempdir)
        bibfile.to_csv(_bibfile, sep=",", index=False)
    except AttributeError:
        _bibfile = bibfile

    data, _, log = read_bib(_bibfile,
                            db=db,
                            del_duplicated=del_duplicated)

    if print_log:
        for line in log:
            print(line)

    return data[list(DEFAULT_FIELDS)]


def generate_oa_bib(bibfile):

    try:
        dataframe = pd.read_csv(bibfile,
                                low_memory=False,
                                lineterminator='\n')
    except TypeError:
        dataframe = bibfile.copy()

    dataframe.rename(FROM_OA,
                     inplace=True,
                     axis="columns")

    return dataframe
