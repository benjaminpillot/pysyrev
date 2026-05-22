import os
import sys

# Make the project root importable so autodoc can find the package
# without it being installed (RTD installs only docs/requirements.txt).
sys.path.insert(0, os.path.abspath('..'))

# =============================================================================
# Project information
# =============================================================================

project = 'pysyrev'
copyright = '2026, Benjamin Pillot'
author = 'Benjamin Pillot'
release = '0.1.0'

# =============================================================================
# Extensions
# =============================================================================

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # NumPy / Google docstrings
    'sphinx.ext.viewcode',       # [source] links
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
}

# =============================================================================
# Mock heavy ML / NLP dependencies that are not installed on RTD
# =============================================================================

autodoc_mock_imports = [
    'bertopic',
    'hdbscan',
    'umap',
    'sentence_transformers',
    'nest_asyncio',
    'litellm',
    'reportlab',
    'plotly',
    'kaleido',
    'langdetect',
    'contractions',
    'bs4',
    'crossref',
    'semanticscholar',
    'nltk',
    'spacy',
    'gensim',
    'octis',
    'sklearn',
]

# =============================================================================
# autodoc defaults
# =============================================================================

autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
    'member-order':     'bysource',
}

autodoc_typehints = 'description'
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# =============================================================================
# HTML output
# =============================================================================

html_theme = 'furo'

html_theme_options = {
    'sidebar_hide_name': False,
}

# =============================================================================
# General
# =============================================================================

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']
