from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Known-garbage phrases (exact match after normalization).
# =============================================================================

# Normalization for exact-match lookup: lowercase, collapse whitespace, strip
# surrounding brackets and punctuation.
_NORMALIZE_RE = re.compile(r'\s+')
_STRIP_CHARS = ' .,;:!?()[]{}"\'\u00a0\t\n\r'

KNOWN_GARBAGE = frozenset({
    '',
    'international audience',
    'no abstract available',
    'no abstract',
    'abstract not available',
    'abstract not provided',
    'abstract unavailable',
    'not available',
    'n/a',
    'na',
    'none',
    'null',
    'unknown',
    'tbd',
    'to be determined',
    'see above',
    'see text',
    'see article',
    'not applicable',
    'summary not available',
})


def _normalize_for_exact_match(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding noise."""
    t = _NORMALIZE_RE.sub(' ', text).strip(_STRIP_CHARS).lower()
    return t


# =============================================================================
# Individual signals. Each returns True when the abstract looks suspicious.
# =============================================================================

# Common English function words / auxiliaries — their presence is a strong
# positive signal that the text is a real English sentence.
_ENGLISH_FUNCTION_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these',
    'those', 'we', 'our', 'they', 'their', 'it', 'its', 'of', 'in', 'on',
    'at', 'by', 'for', 'with', 'from', 'to', 'and', 'or', 'but', 'not',
    'as', 'than', 'then', 'which', 'while', 'between', 'among', 'however',
    'thus', 'therefore', 'moreover', 'furthermore', 'where', 'when',
})

# Verb cues frequently found in scientific abstracts.
_VERB_CUES = frozenset({
    'show', 'shows', 'showed', 'present', 'presents', 'presented',
    'propose', 'proposes', 'proposed', 'demonstrate', 'demonstrates',
    'demonstrated', 'find', 'finds', 'found', 'describe', 'describes',
    'described', 'report', 'reports', 'reported', 'investigate',
    'investigates', 'investigated', 'study', 'studies', 'studied',
    'analyze', 'analyzes', 'analyzed', 'analyse', 'analyses', 'analysed',
    'develop', 'develops', 'developed', 'examine', 'examines', 'examined',
    'suggest', 'suggests', 'suggested', 'provide', 'provides', 'provided',
    'use', 'uses', 'used', 'apply', 'applies', 'applied',
})

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_SENTENCE_TERMINATOR_RE = re.compile(r'[.!?]')


def clean_doi(doi):
    valid_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ./-_:')

    # if 'unknown' in doi.lower():
    #     return ""

    try:
        cleaned_doi = ''
        for char in doi:
            if char in valid_chars:
                cleaned_doi = cleaned_doi + char
            else:
                break
    except TypeError:
        cleaned_doi = doi

    return cleaned_doi


def _signal_no_function_words(words_lower: list[str]) -> bool:
    """True if the text contains zero English function words.
    A real English sentence has several; a keyword list has none."""
    return not any(w in _ENGLISH_FUNCTION_WORDS for w in words_lower)


def _signal_no_verbs(words_lower: list[str]) -> bool:
    """True if the text contains no common scientific-abstract verb.
    Catches keyword lists and technical-term dumps."""
    return not any(w in _VERB_CUES for w in words_lower)


def _signal_high_lexical_diversity(words_lower: list[str]) -> bool:
    """True if almost every word is unique. Real prose repeats function
    words heavily (type-token ratio typically < 0.75)."""
    if len(words_lower) < 6:
        return False  # too short to assess reliably, skip
    ratio = len(set(words_lower)) / len(words_lower)
    return ratio > 0.85


def _signal_no_sentence_structure(text: str) -> bool:
    """True if the text has no sentence-ending punctuation at all.
    Keyword lists and fragments typically lack one."""
    return not _SENTENCE_TERMINATOR_RE.search(text)


def _signal_non_english(text: str, detector) -> bool:
    """True if langdetect is confident the text is not English."""
    if detector is None:
        return False
    try:
        # detect_langs returns ranked candidates with probabilities.
        langs = detector(text)
        if not langs:
            return False
        top = langs[0]
        return top.lang != 'en' and top.prob > 0.9
    except Exception:
        return False


def is_plausible_abstract(
    text,
    min_signals_to_reject: int,
    extra_garbage_phrases: Iterable[str],
    detector,
) -> bool:
    """
    Return True if `text` looks like a plausible English scientific abstract.

    Parameters
    ----------
    text : str or NaN
        Candidate abstract. NaN / None / empty returns False (not plausible).
    min_signals_to_reject : int, default 2
        Number of suspicion signals required to reject. 2 = permissive,
        3 = very permissive, 1 = strict.
    extra_garbage_phrases : iterable of str
        Additional phrases to treat as known-garbage (normalized exact match).
    detector : callable or None
        Optional langdetect-style callable returning a ranked list of
        (lang, prob) candidates. Pass `langdetect.detect_langs` to enable
        the language check. When None, the language signal is skipped.

    Returns
    -------
    bool
        True if the abstract passes the relevance checks.
    """
    # 1. Missing value.
    if not isinstance(text, str) or not text.strip():
        return False

    # 2. Exact match against known garbage phrases.
    normalized = _normalize_for_exact_match(text)
    garbage = KNOWN_GARBAGE.union(
        _normalize_for_exact_match(p) for p in extra_garbage_phrases
    )
    if normalized in garbage:
        return False

    # 3. Multi-signal vote.
    words_lower = [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    suspicious_signals = sum([
        _signal_no_function_words(words_lower),
        _signal_no_verbs(words_lower),
        _signal_high_lexical_diversity(words_lower),
        _signal_no_sentence_structure(text),
        _signal_non_english(text, detector),
    ])

    return suspicious_signals < min_signals_to_reject


def clean_abstracts(
    series: pd.Series,
    min_signals_to_reject: int,
    extra_garbage_phrases: Iterable[str],
    use_langdetect: bool,
) -> pd.Series:
    """
    Vectorized wrapper over `is_plausible_abstract`. Returns a Series where
    implausible abstracts are replaced by NaN.

    Parameters
    ----------
    series : pd.Series
        The abstract column.
    min_signals_to_reject : int, default 2
        See `is_plausible_abstract`.
    extra_garbage_phrases : iterable of str
        See `is_plausible_abstract`.
    use_langdetect : bool, default True
        If True and the `langdetect` package is importable, non-English
        abstracts with high confidence are flagged. If False or import fails,
        the language signal is skipped.
    """
    detector = None
    if use_langdetect:
        try:
            from langdetect import detect_langs, DetectorFactory
            # Deterministic seed: same input -> same output across runs.
            DetectorFactory.seed = 0
            detector = detect_langs
        except ImportError:
            pass

    plausible = series.map(
        lambda t: is_plausible_abstract(
            t, min_signals_to_reject, extra_garbage_phrases, detector,
        )
    )
    return series.where(plausible, other=np.nan)
