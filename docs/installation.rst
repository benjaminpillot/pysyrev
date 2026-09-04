Installation
============

Prerequisites
-------------

- Python ≥ 3.10
- A virtual environment is strongly recommended (conda or venv)

From PyPI
---------

.. code-block:: bash

   pip install pysyrev

From source
-----------

.. code-block:: bash

   git clone <repo-url>
   cd pysyrev
   pip install -e .

The editable install (``-e``) lets you update the package by pulling from
the repository without reinstalling.

spaCy model
-----------

Topic-model pre-processing (lemmatisation) needs a spaCy English model, which
is not a pip dependency. Install it once:

.. code-block:: bash

   python -m spacy download en_core_web_lg

Optional extras
---------------

.. code-block:: bash

   pip install "pysyrev[citations]"   # BibDataset.fetch_citations (CrossRef, Semantic Scholar)
   pip install "pysyrev[langdetect]"  # bib.clean.use_langdetect

Plotly and kaleido, which render the report figures, are installed by default —
the ``[plotly]`` extra is kept only so older instructions keep working.

API keys
--------

pysyrev reads credentials from a ``.env`` file. Point to it with the root-level
``env:`` key in your config YAML:

.. code-block:: yaml

   env: /path/to/.env

.. code-block:: bash
   :caption: .env

   WOS_API_KEY=your-wos-key
   OPENALEX_API_KEY=your-openalex-key
   ANTHROPIC_API_KEY=your-anthropic-key
   OPENAI_API_KEY=your-openai-key
   ALBERT_API_KEY=your-albert-key
   UNPAYWALL_EMAIL=you@example.org
   ELSEVIER_API_KEY=your-elsevier-key

Any ``${VAR}`` reference in the YAML is resolved against these variables at
load time. Variables already set in the process environment take precedence.
