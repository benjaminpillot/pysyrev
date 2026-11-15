"""
Report generation: the PDF and HTML engines, their figures, and the sections
they render.

The engines (:mod:`~pysyrev.core.report.engine`, :mod:`~pysyrev.core.report.html`)
know nothing about bibliometrics — they render a declarative ``report_data``
dict of typed blocks.
:func:`~pysyrev.core.report.data.build_report_data` is what produces that dict
from one topic-model run, delegating each section to
:mod:`pysyrev.core.report.sections`.

Layout
------
:mod:`~pysyrev.core.report.engine`
    The ReportLab engine: theme, layout, and one renderer per block type.
:mod:`~pysyrev.core.report.html`
    The HTML engine: the same blocks as one self-contained interactive page.
:mod:`~pysyrev.core.report.live`
    That page served locally, with a panel of settings that rebuild a section.
:mod:`~pysyrev.core.report.figures`
    Plotly figures shared across sections (the matrix heatmaps).
:mod:`~pysyrev.core.report.data`
    Run loading, and the assembly of the whole ``report_data`` dict.
:mod:`~pysyrev.core.report.sections`
    One module per section: topics, networks, temporal, papers.
"""

from pysyrev.core.report.engine import (  # noqa: F401
    PDFReportEngine, PlotlyAssetManager, ReportLayout, ReportTheme,
    build_resized_image, build_styles, build_table, pdf_text, safe_str,
    validate_report_data)
from pysyrev.core.report.html import HTMLReportEngine, html_text  # noqa: F401
from pysyrev.core.report.data import (  # noqa: F401
    build_file_prefix, build_report_data, find_best_results_csv,
    load_bertopic_results, load_topic_info)
