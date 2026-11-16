Report engines
==============

The report is first built as a declarative ``report_data`` dictionary — a
``meta`` block and a list of sections, each a list of typed blocks (paragraph,
table, plotly figure, callout…). An engine then renders that dictionary: the
same data gives the PDF, the HTML page and the live report. See
:doc:`../report` for the user-facing side.

A block may carry ``"only": "pdf"`` or ``"only": "html"`` to be rendered by one
engine and skipped by the other.

Building the data
-----------------

.. autofunction:: pysyrev.core.report.build_report_data
.. autofunction:: pysyrev.core.report.validate_report_data

PDF
---

.. autoclass:: pysyrev.core.report.PDFReportEngine
   :members:

HTML
----

.. autoclass:: pysyrev.core.report.HTMLReportEngine
   :members:

.. autofunction:: pysyrev.core.report.html_text

Live report
-----------

.. automodule:: pysyrev.core.report.live
   :no-members:

.. autoclass:: pysyrev.core.report.live.LiveReport
   :members:

The settings the panel exposes are listed in ``pysyrev.core.report.live.CONTROLS``
— see the table in :doc:`../report`.
