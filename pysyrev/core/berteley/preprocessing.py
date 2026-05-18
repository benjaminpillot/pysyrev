import csv
import re
import string
import subprocess
from functools import partial
from pathlib import Path
from typing import List

import nltk
from bs4 import BeautifulSoup
import contractions
from tqdm import tqdm


# =============================================================================
# Lazy loading of heavy resources.
#
# spacy.load('en_core_web_lg') loads ~700 MB into memory and en_core_web_lg
# runs a network check at import time when missing. All loading is deferred
# until the first call to a function that actually needs the resource.
# =============================================================================

_NLP = None
_STOPWORDS = None
_NLTK_STOPWORDS = None


def _ensure_nltk():
    for resource, path in (('stopwords', 'corpora/stopwords'),
                           ('punkt_tab', 'tokenizers/punkt_tab')):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def _ensure_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy
    try:
        _NLP = spacy.load('en_core_web_lg')
    except OSError:
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_lg'],
                       check=True)
        _NLP = spacy.load('en_core_web_lg')
    return _NLP


def _ensure_stopwords():
    global _STOPWORDS, _NLTK_STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS, _NLTK_STOPWORDS
    _ensure_nltk()
    from nltk.corpus import stopwords
    here = Path(__file__).parent
    with open(here / 'berteley_stopwords.csv', 'r') as f:
        _STOPWORDS = [row[0] for row in csv.reader(f, delimiter=',')]
    _NLTK_STOPWORDS = set(stopwords.words('english'))
    return _STOPWORDS, _NLTK_STOPWORDS


# =============================================================================
# Per-document transforms
# =============================================================================

def combine_hyphens(doc: str) -> str:
    row_string = re.sub('-', '', doc)
    return ' '.join(row_string.split())


def remove_punctuation(doc: str) -> str:
    return doc.translate(str.maketrans('', '', string.punctuation))


def lemmatize(doc: str) -> str:
    nlp = _ensure_spacy()
    return ' '.join(word.lemma_ for word in nlp(doc) if word.lemma_ != '-PRON-')


def remove_stopwords(doc: str, allow_abbrev: bool = True) -> str:
    _ensure_nltk()
    from nltk.tokenize import word_tokenize
    berteley_stopwords, nltk_stopwords = _ensure_stopwords()
    filt = []
    for word in word_tokenize(doc):
        if (re.search('[a-zA-Z]', word) is not None
                and word.lower() not in nltk_stopwords
                and word.lower() not in berteley_stopwords):
            if allow_abbrev or len(word.lower()) > 2:
                filt.append(word)
    return ' '.join(filt)


def remove_html(doc: str) -> str:
    return BeautifulSoup(doc, 'html.parser').get_text(separator=' ')


def remove_extraspace(doc: str) -> str:
    return ' '.join(doc.split())


def expand_contractions(doc: str) -> str:
    return contractions.fix(doc)


# =============================================================================
# Preprocessing pipeline
# =============================================================================

def preprocess(docs: List[str], allow_abbrev: bool = True,
               show_progress: bool = False) -> List[str]:
    """Run the full preprocessing pipeline on a list of raw documents.

    Steps: HTML stripping → contraction expansion → lowercasing →
    punctuation removal → whitespace normalization → lemmatization →
    stopword removal → short-document filtering (≤ 10 tokens dropped).
    """
    cleaned = docs.copy()

    steps = [
        ('Removing HTML tags',        remove_html,                                   None),
        ('Expanding contractions',    expand_contractions,                            None),
        ('Lowercasing',               str.lower,                                      None),
        ('Removing punctuation',      remove_punctuation,                             None),
        ('Normalising whitespace',    remove_extraspace,                              None),
        ('Lemmatizing',               lemmatize,                                      None),
        ('Removing stopwords',        partial(remove_stopwords, allow_abbrev=allow_abbrev), None),
        ('Filtering short documents', None,                                           lambda d: len(d.split()) > 10),
    ]

    iterator = tqdm(steps, desc='Preprocessing') if show_progress else steps
    for label, func, filt in iterator:
        if show_progress:
            iterator.set_description(label)
        if func is not None:
            cleaned = [func(d) for d in cleaned]
        if filt is not None:
            cleaned = [d for d in cleaned if filt(d)]

    return cleaned
