"""
Tests for pysyrev.core.report.html (HTMLReportEngine) and for the report
formats TopicReport writes.

Fast: no ML dependency, plotly only where a figure is involved.
"""

import json
import re
from types import SimpleNamespace

import pytest

from pysyrev.core.config import TopicReportExportConfig
from pysyrev.core.report import HTMLReportEngine, PDFReportEngine, html_text


def _page(blocks, **meta):
    data = {"meta": {"title": "T", **meta},
            "sections": [{"title": "1. Section", "blocks": blocks}]}
    return HTMLReportEngine().render(data)


# =============================================================================
# Text
# =============================================================================

class TestHtmlText:
    def test_escapes_what_is_not_markup(self):
        assert html_text("a < b & c") == "a &lt; b &amp; c"
        assert "<script>" not in html_text("<script>alert(1)</script>")

    def test_restores_the_inline_markup(self):
        assert html_text("<b>Bold</b> and <i>it</i><br/>next") == \
            "<b>Bold</b> and <i>it</i><br>next"

    def test_reportlab_link_becomes_an_anchor(self):
        out = html_text('<link href="https://x.org/a" color="#0B63CE">here</link>')
        assert out == '<a href="https://x.org/a" target="_blank" rel="noopener">here</a>'

    def test_unsafe_href_keeps_only_the_label(self):
        assert html_text('<link href="javascript:alert(1)">x</link>') == "x"

    def test_none_is_empty(self):
        assert html_text(None) == ""


# =============================================================================
# Page
# =============================================================================

class TestRender:
    def test_page_carries_every_block(self, minimal_report_data):
        page = HTMLReportEngine().render(minimal_report_data)
        assert page.startswith("<!DOCTYPE html>")
        for expected in ("Test Report", "Automated test suite", "Hello world.",
                         "<b>Bold</b>", "<li>Bullet A</li>", "<li>First</li>",
                         "<b>Key :</b> Value", "A test callout.",
                         "x = 1 &lt; 2", "Nested paragraph.", "Pytest"):
            assert expected in page

    def test_sections_and_subsections_are_in_the_sidebar(self, minimal_report_data):
        page = HTMLReportEngine().render(minimal_report_data)
        assert 'data-section="summary"' in page          # meta.summary → a panel
        assert '<a href="#section-1" data-section="section-1">Section 1</a>' in page
        assert '<div class="subnav" data-for="section-1">' in page
        assert '<a href="#section-1-1">Sub</a>' in page
        assert '<h3 id="section-1-1">Sub</h3>' in page

    def test_no_plotlyjs_without_a_figure(self, minimal_report_data):
        assert len(HTMLReportEngine().render(minimal_report_data)) < 100_000

    def test_validation_is_shared_with_the_pdf_engine(self):
        with pytest.raises(ValueError, match="meta"):
            HTMLReportEngine().render({"sections": []})

    def test_unknown_block_type_is_reported_not_fatal(self):
        assert "Unsupported block type: nope" in _page([{"type": "nope"}])

    def test_generate_writes_the_file(self, minimal_report_data, tmp_path):
        out = tmp_path / "r.html"
        HTMLReportEngine().generate(minimal_report_data, str(out))
        assert "Hello world." in out.read_text(encoding="utf-8")

    def test_rejects_an_unknown_plotlyjs_mode(self):
        with pytest.raises(ValueError):
            HTMLReportEngine(plotlyjs="nope")


class TestTable:
    BLOCK = {"type": "table", "title": "Papers", "headers": ["Title", "DOI"],
             "rows": [["A <i>study</i> of x < y", "10.1000/abc.1"],
                      ["Other", "https://x.org/p"]],
             "col_widths": [3.0, 1.0], "caption": "cap"}

    def test_structure(self):
        page = _page([self.BLOCK])
        assert "<h4>Papers</h4>" in page
        assert "<th>Title</th><th>DOI</th>" in page
        assert '<col style="width:75.0%">' in page
        assert "A <i>study</i> of x &lt; y" in page
        assert '<div class="caption">cap</div>' in page

    def test_doi_and_url_cells_are_links(self):
        page = _page([self.BLOCK])
        assert 'href="https://doi.org/10.1000/abc.1"' in page
        assert 'href="https://x.org/p"' in page

    def test_filter_box_only_on_long_tables(self):
        assert 'type="search"' not in _page([self.BLOCK])
        long = dict(self.BLOCK, rows=[["t", "d"]] * 10)
        assert 'type="search"' in _page([long])


class TestOnly:
    BLOCKS = [{"type": "paragraph", "text": "for-the-pdf", "only": "pdf"},
              {"type": "paragraph", "text": "for-the-page", "only": "html"},
              {"type": "paragraph", "text": "for-both"}]

    def test_html_engine_skips_pdf_only_blocks(self):
        page = _page(self.BLOCKS)
        assert "for-the-pdf" not in page
        assert "for-the-page" in page and "for-both" in page

    def test_pdf_engine_skips_html_only_blocks(self):
        engine = PDFReportEngine()
        assert engine._render_block(self.BLOCKS[1]) == []
        assert len(engine._render_block(self.BLOCKS[0])) == 1
        assert len(engine._render_block(self.BLOCKS[2])) == 1


class TestPlotly:
    @pytest.fixture
    def figure(self):
        go = pytest.importorskip("plotly.graph_objects")
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4],
                                   text=["a</script><b>x", "b"]))
        fig.update_layout(width=850, height=450)
        return fig

    def _spec(self, page):
        raw = re.search(r'<script type="application/json" class="figure-spec">'
                        r'(.*?)</script>', page, re.S).group(1)
        return raw, json.loads(raw)

    def test_figure_is_embedded_live(self, figure):
        page = HTMLReportEngine(plotlyjs="cdn").render({
            "meta": {"title": "T"},
            "sections": [{"title": "S", "blocks": [
                {"type": "plotly", "figure": figure, "caption": "the caption"}]}]})
        raw, spec = self._spec(page)
        assert spec["data"][0]["type"] == "scatter"
        # Sized for the page, not for the PDF's PNG.
        assert "width" not in spec["layout"]
        assert spec["layout"]["height"] == 450 and spec["layout"]["autosize"] is True
        assert 'class="figure-plot"' in page and "the caption" in page
        assert "cdn.plot.ly/plotly-" in page

    def test_a_script_tag_in_the_data_cannot_close_the_element(self, figure):
        page = HTMLReportEngine(plotlyjs="cdn").render({
            "meta": {"title": "T"},
            "sections": [{"title": "S", "blocks": [
                {"type": "plotly", "figure": figure}]}]})
        raw, spec = self._spec(page)
        assert "<" not in raw
        assert spec["data"][0]["text"][0] == "a</script><b>x"

    def test_the_caller_figure_is_left_untouched(self, figure):
        _page([{"type": "plotly", "figure": figure}])
        assert figure.layout.width == 850

    def test_embed_inlines_plotlyjs(self, figure):
        page = _page([{"type": "plotly", "figure": figure}])
        assert "<script src=" not in page
        assert len(page) > 1_000_000


# =============================================================================
# Formats
# =============================================================================

class TestFormatsConfig:
    def test_default_is_both(self):
        assert TopicReportExportConfig(export_dir="x").formats == ["pdf", "html"]

    def test_blank_falls_back_to_both(self):
        assert TopicReportExportConfig(export_dir="x", formats=None).formats == ["pdf", "html"]

    def test_a_single_string_is_accepted(self):
        assert TopicReportExportConfig(export_dir="x", formats="html").formats == ["html"]

    @pytest.mark.parametrize("bad", [["docx"], []])
    def test_rejects_unknown_or_empty(self, bad):
        with pytest.raises(ValueError, match="formats"):
            TopicReportExportConfig(export_dir="x", formats=bad)


class TestGenerateReportFormats:
    """generate_report writes one file per format, from one report_data."""

    def _report(self, monkeypatch, minimal_report_data, formats):
        import pysyrev.topic_report as mod
        monkeypatch.setattr(mod, "build_report_data",
                            lambda *a, **k: minimal_report_data)
        report = mod.TopicReport(
            run_dir="unused",
            report_config=SimpleNamespace(
                export=SimpleNamespace(formats=formats) if formats else None))
        for attr in ("_best_results", "_topic_info", "_bertopic_results"):
            setattr(report, attr, object())
        return report

    def test_both_by_default(self, monkeypatch, minimal_report_data, tmp_path):
        report = self._report(monkeypatch, minimal_report_data, None)
        out = report.generate_report(str(tmp_path / "r.pdf"))
        assert out == str(tmp_path / "r.pdf")
        assert (tmp_path / "r.pdf").read_bytes().startswith(b"%PDF")
        assert "Hello world." in (tmp_path / "r.html").read_text(encoding="utf-8")

    def test_html_only(self, monkeypatch, minimal_report_data, tmp_path):
        report = self._report(monkeypatch, minimal_report_data, ["html"])
        out = report.generate_report(str(tmp_path / "r.pdf"))
        assert out == str(tmp_path / "r.html")
        assert not (tmp_path / "r.pdf").exists()

    def test_a_failing_page_does_not_cost_the_pdf(self, monkeypatch,
                                                  minimal_report_data, tmp_path, capsys):
        import pysyrev.topic_report as mod

        def _boom(self, *a, **k):
            raise RuntimeError("no page")

        monkeypatch.setattr(mod.HTMLReportEngine, "generate", _boom)
        report = self._report(monkeypatch, minimal_report_data, ["pdf", "html"])
        out = report.generate_report(str(tmp_path / "r.pdf"))
        assert out == str(tmp_path / "r.pdf")
        assert "HTML report skipped" in capsys.readouterr().out

    def test_a_failing_page_is_fatal_on_its_own(self, monkeypatch,
                                                minimal_report_data, tmp_path):
        import pysyrev.topic_report as mod

        def _boom(self, *a, **k):
            raise RuntimeError("no page")

        monkeypatch.setattr(mod.HTMLReportEngine, "generate", _boom)
        report = self._report(monkeypatch, minimal_report_data, ["html"])
        with pytest.raises(RuntimeError):
            report.generate_report(str(tmp_path / "r.pdf"))
