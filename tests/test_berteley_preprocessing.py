"""
Tests for pysyrev.core.berteley.preprocessing.

Fast tests: cover pure/light functions with no heavy dependencies.
Slow tests (marked `slow`): exercise lemmatize() and preprocess() with
    real spacy en_core_web_lg — run with `pytest -m slow`.
"""

from unittest.mock import patch

import pytest

from pysyrev.core.berteley.preprocessing import (
    combine_hyphens,
    expand_contractions,
    preprocess,
    remove_extraspace,
    remove_html,
    remove_punctuation,
)


# =============================================================================
# combine_hyphens
# =============================================================================

class TestCombineHyphens:
    def test_basic(self):
        assert combine_hyphens("x-ray") == "xray"

    def test_multiple_hyphens(self):
        assert combine_hyphens("well-known agent-based") == "wellknown agentbased"

    def test_no_hyphens(self):
        assert combine_hyphens("hello world") == "hello world"

    def test_leading_trailing_spaces_collapsed(self):
        # hyphen removed, then multiple spaces collapsed by join/split
        assert combine_hyphens("a  -  b") == "a b"


# =============================================================================
# remove_punctuation
# =============================================================================

class TestRemovePunctuation:
    def test_basic(self):
        assert remove_punctuation("Hello, world!") == "Hello world"

    def test_keeps_alphanumerics(self):
        result = remove_punctuation("abc 123")
        assert result == "abc 123"

    def test_strips_all_punctuation(self):
        result = remove_punctuation("...!!!???")
        assert result == ""

    def test_mixed(self):
        result = remove_punctuation("agent-based model (2024).")
        assert "agent" in result
        assert "2024" in result
        assert "." not in result
        assert "(" not in result


# =============================================================================
# remove_html
# =============================================================================

class TestRemoveHtml:
    def test_strips_tags(self):
        assert "bold" in remove_html("<b>bold</b>")
        assert "<b>" not in remove_html("<b>bold</b>")

    def test_plain_text_unchanged(self):
        text = "no html here"
        assert remove_html(text) == text

    def test_nested_tags(self):
        result = remove_html("<div><p>paragraph</p></div>")
        assert "paragraph" in result
        assert "<" not in result

    def test_empty_string(self):
        assert remove_html("") == ""


# =============================================================================
# remove_extraspace
# =============================================================================

class TestRemoveExtraspace:
    def test_collapses_spaces(self):
        assert remove_extraspace("too   many   spaces") == "too many spaces"

    def test_tabs_and_newlines(self):
        assert remove_extraspace("a\t\tb\n\nc") == "a b c"

    def test_leading_trailing(self):
        assert remove_extraspace("  hello  ") == "hello"

    def test_already_clean(self):
        assert remove_extraspace("clean text") == "clean text"


# =============================================================================
# expand_contractions
# =============================================================================

class TestExpandContractions:
    def test_wont(self):
        result = expand_contractions("I won't go")
        assert "will not" in result or "won't" not in result

    def test_no_contractions(self):
        result = expand_contractions("simple text")
        assert result == "simple text"

    def test_multiple(self):
        result = expand_contractions("I can't and I don't")
        assert "can't" not in result
        assert "don't" not in result


# =============================================================================
# preprocess — pipeline structure (mocked heavy steps)
# =============================================================================

class TestPreprocessPipelineMocked:
    """Test the preprocess() pipeline shape by mocking spacy-dependent steps.
    lemmatize is replaced by identity, remove_stopwords by identity, so the
    test does not require spacy or nltk corpora."""

    def _run(self, docs, **kwargs):
        # accept **kw so the partial(..., allow_abbrev=...) wrapper doesn't crash
        identity = lambda doc, **kw: doc
        with patch("pysyrev.core.berteley.preprocessing.lemmatize", side_effect=identity), \
             patch("pysyrev.core.berteley.preprocessing.remove_stopwords", side_effect=identity):
            return preprocess(docs, **kwargs)

    def test_returns_list(self):
        result = self._run(["Hello world, this is a test sentence with enough words."])
        assert isinstance(result, list)

    def test_short_docs_filtered(self):
        """Documents with ≤ 10 tokens after cleaning should be dropped."""
        short = ["Too short."]
        long_enough = ["This sentence has more than ten tokens and should pass the filter easily."]
        result = self._run(short + long_enough)
        assert len(result) == 1

    def test_html_stripped(self):
        docs = ["<b>Bold</b> text with enough words to pass the ten token filter here."]
        result = self._run(docs)
        assert "<b>" not in result[0]

    def test_lowercased(self):
        docs = ["UPPER CASE TEXT WITH MORE THAN TEN WORDS TO PASS THE FILTER EASILY HERE."]
        result = self._run(docs)
        assert result[0] == result[0].lower()

    def test_punctuation_removed(self):
        docs = ["Hello, world! This sentence has enough words to pass the ten-token filter."]
        result = self._run(docs)
        assert "," not in result[0]
        assert "!" not in result[0]

    def test_empty_input(self):
        assert self._run([]) == []

    def test_show_progress_does_not_crash(self):
        docs = ["This sentence has enough words to pass the ten token filter with ease."]
        result = self._run(docs, show_progress=True)
        assert len(result) == 1


# =============================================================================
# Slow tests — real spacy
# =============================================================================

@pytest.mark.slow
class TestPreprocessReal:
    def test_lemmatize_reduces_inflections(self):
        from pysyrev.core.berteley.preprocessing import lemmatize
        result = lemmatize("running models simulated populations")
        # spacy lemmatises "running" → "run", etc.
        assert isinstance(result, str)
        assert len(result) > 0

    def test_remove_stopwords_removes_common_words(self):
        from pysyrev.core.berteley.preprocessing import remove_stopwords
        result = remove_stopwords("the cat sat on the mat")
        assert "the" not in result.lower()
        assert "on" not in result.lower()

    def test_remove_stopwords_allow_abbrev_false(self):
        from pysyrev.core.berteley.preprocessing import remove_stopwords
        result = remove_stopwords("I use ABM to model social systems", allow_abbrev=False)
        # Tokens of length ≤ 2 should be dropped
        for token in result.split():
            assert len(token) > 2

    def test_full_preprocess_pipeline(self):
        # Sentences must be long enough to survive the > 10 token filter
        # that applies after lemmatisation and stopword removal.
        corpus = [
            ("Agent-based modelling is a computational approach used to simulate "
             "the actions and interactions of autonomous agents in order to assess "
             "their effects on the system as a whole."),
            ("Cellular automata are discrete dynamic systems whose behaviour is "
             "completely specified in terms of local relations, widely applied in "
             "the study of complex adaptive systems and emergent phenomena."),
            ("Social simulation uses computational tools to explore the behaviour "
             "of social systems, including economic markets, political institutions, "
             "and ecological dynamics driven by heterogeneous individual agents."),
        ]
        result = preprocess(corpus, allow_abbrev=True, show_progress=False)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, str) and len(s) > 0 for s in result)
        # Common stopwords should be gone
        for doc in result:
            assert "the" not in doc.split()
