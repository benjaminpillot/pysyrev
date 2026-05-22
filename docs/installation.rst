Installation
============

Prerequisites
-------------

- Python ≥ 3.10
- A virtual environment is strongly recommended (conda or venv)

From source
-----------

.. code-block:: bash

   git clone <repo-url>
   cd pysyrev
   pip install -r requirements.txt
   pip install -e .

The editable install (``-e``) lets you update the package by pulling from
the repository without reinstalling.

Optional extras
---------------

To enable Plotly figure embedding in PDF reports:

.. code-block:: bash

   pip install -e ".[plotly]"

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

Any ``${VAR}`` reference in the YAML is resolved against these variables at
load time.
