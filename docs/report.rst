The report: PDF, HTML and live mode
===================================

The ``topic-report`` stage turns one topic-model run into a report. The same
content is available in three forms:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Form
     - How to get it
     - What it is for
   * - PDF
     - ``pysyrev config.yaml`` (default)
     - A printable, paginated document with static figures — the version to
       archive or attach to a paper.
   * - HTML page
     - ``pysyrev config.yaml`` (default)
     - One self-contained file to open in a browser: interactive figures,
       sortable tables, clickable DOIs.
   * - Live mode
     - ``pysyrev serve config.yaml``
     - The HTML page served locally with a settings panel: change a report
       setting and see its section redrawn at once.

Both files are written by default. Choose with ``topic_report.export.formats``:

.. code-block:: yaml

   topic_report:
     export:
       formats: [pdf, html]   # or [pdf], or [html]

They land side by side in the report's run directory, under the same name:

.. code-block:: text

   <output_dir>/topic_report/report_<timestamp>/
   ├── topic_report_<timestamp>.pdf
   ├── topic_report_<timestamp>.html
   └── …                                 # network figures, paper-selection annex

If the HTML export fails while a PDF was asked for, the PDF is kept and the
failure is reported — the run does not stop.


The HTML page
-------------

The page is a single ``.html`` file: no server, no companion folder. Send it by
e-mail or drop it on a shared drive and it opens as is.

- **Navigation** — a sidebar lists the sections and their subsections; one
  section is shown at a time. The address carries the current section, so a
  link to the page can point at a given section.
- **Figures** — every plotly figure is live: hover for the details (for a
  network node: the paper, its community and topic; for an edge: the number of
  shared references or co-citing documents), zoom, pan, and click a legend
  entry to hide or isolate a community or a topic.
- **Tables** — click a column header to sort. Tables of ten rows or more get a
  filter box.
- **Links** — DOIs and URLs found in table cells are turned into links, so a
  paper of the reading list is one click away.

.. note::

   plotly.js is embedded in the page, which is what lets it work offline. It
   weighs about 5 MB, so the file is that large whatever the size of the
   report.

A few blocks differ between the two formats: the PDF's pointers to the
"interactive version" of a network figure are left out of the HTML page, where
the figure itself is interactive.


Live mode
---------

Many report settings only change how a figure or a table is *drawn*: how many
actors a ranking lists, what the node size stands for, how many edges are
kept on a network. Live mode lets you try them without regenerating the report
each time.

.. code-block:: bash

   pysyrev serve config.yaml
   pysyrev serve config.yaml --port 9000 --no-browser

The command reads the latest topic-model run (or ``topic_report.run_dir``),
builds the report in memory and opens it in the browser, at
``http://127.0.0.1:8765`` by default. If the port is taken, the next free one
is used and printed. Stop the server with :kbd:`Ctrl+C`.

The page is the HTML report with a settings panel beside it. The panel shows
the settings of the section on screen:

- changing a value rebuilds **that section only** and swaps it into the page —
  the rest of the report is untouched;
- a value the config would refuse (an unknown option, an out-of-range number)
  is refused in the panel too, with the config's own error message;
- **Reset** returns the section to the values of the config file.

Nothing is written to disk — neither a report nor the config. To keep a
setting, copy the YAML snippet the panel shows and paste it into your config:

.. code-block:: yaml

   topic_report:
     sections:
       actors:
         top_n: 25
         network:
           size_by: papers

Then run ``pysyrev config.yaml --stage topic-report`` to write the final PDF
and HTML files with those values.

Settings available in the panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only settings that are cheap to apply are exposed: those that redraw a table
or a figure without recomputing a network. They are documented, with their
defaults, in the :doc:`configuration reference <configuration>`.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Section
     - Setting
     - Effect
   * - ``actors``
     - ``levels``
     - Levels shown: country, institution, author
   * -
     - ``top_n``
     - Actors listed per ranking table
   * -
     - ``sort_by``
     - Ranking criterion: papers, fractional or citations
   * -
     - ``network.size_by``
     - What the node area stands for
   * -
     - ``network.size_range``
     - Smallest / largest node diameter
   * -
     - ``network.edge_width_by``
     - What the edge width stands for
   * -
     - ``network.edge_width_range``
     - Thinnest / thickest edge
   * -
     - ``network.backbone_k``
     - Strongest links drawn per node
   * -
     - ``network.n_labels``
     - Number of nodes named on the graph
   * -
     - ``network.node_opacity``
     - Whether edges show through the nodes
   * - ``bib_network``
     - ``coupling.color_by``, ``cocitation.color_by``
     - Colour the nodes by topic or by community
   * -
     - ``coupling.backbone_k``, ``cocitation.backbone_k``
     - Strongest links drawn per node
   * -
     - ``coupling.edge_scale``, ``cocitation.edge_scale``
     - Edge widths by tertile or linear
   * -
     - ``coupling.edge_width_range``, ``cocitation.edge_width_range``
     - Thinnest / thickest edge (linear scale)
   * -
     - ``connectivity.min_support``
     - Observations needed before an internal ratio is reported

Settings that change *what is computed* — the Leiden resolution sweep, the
minimum community size, the nodes kept in a collaboration network, the
paper-selection rules — are not in the panel: change them in the config and
run the ``topic-report`` stage again.

.. note::

   Changing a drawing option of a collaboration network is quick, because the
   network itself is computed once and kept. Changing a setting of the
   bibliographic networks redraws the whole section, which takes longer on a
   large corpus.

Things to know
~~~~~~~~~~~~~~

- **Local, single user.** The server listens on ``127.0.0.1`` only, so the
  page is reachable from your machine alone. ``--host`` can open it to the
  network, but there is no authentication: do it only on a network you trust.
- **LLM labels.** With an ``llm`` section, ``serve`` uses the topic and
  community labels cached in the run directory by a previous report. If none
  are cached yet, it asks the LLM for them once, as the ``topic-report`` stage
  would. Generate the report once before serving to avoid that call.
- **Reference metadata.** With ``bib_network.cocitation.metadata`` set, the
  co-cited references are resolved through OpenAlex each time the
  bibliographic-network section is built — so at start-up and at every change
  of one of its settings. Give that block a ``cache`` file: only the
  references missing from it are then fetched, and a rebuild makes no request
  at all once the cache is complete.


From Python
-----------

.. code-block:: python

   from pysyrev import TopicReport
   from pysyrev.core.config import Config

   report = TopicReport.from_config(Config.load("config.yaml"))

   report.generate_report()        # writes the configured formats
   report.serve(port=8765)         # live mode, until interrupted

The engines can also be used on their own, on the declarative ``report_data``
dictionary every section is built into — see :doc:`api/report`.
