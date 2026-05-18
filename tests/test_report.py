"""
Tests for pysyrev.core.report (PDFReportEngine, ReportTheme, ReportLayout).

All tests here are fast — no heavy ML dependencies, only reportlab.
"""

from pathlib import Path

import pytest
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, Spacer, Table

from pysyrev.core.report import PDFReportEngine, ReportLayout, ReportTheme


# =============================================================================
# ReportTheme
# =============================================================================

class TestReportTheme:
    def test_defaults(self):
        theme = ReportTheme()
        assert theme.primary == "#16324F"
        assert theme.link == "#0B63CE"

    def test_override(self):
        theme = ReportTheme(primary="#FF0000")
        assert theme.primary == "#FF0000"
        assert theme.secondary == "#2B4C6F"  # unchanged


# =============================================================================
# ReportLayout
# =============================================================================

class TestReportLayout:
    def test_defaults(self):
        lo = ReportLayout()
        assert lo.left_margin_cm == 2.0
        assert lo.right_margin_cm == 2.0
        assert lo.top_margin_cm == 2.4
        assert lo.bottom_margin_cm == 2.2
        assert lo.cover_top_spacer_cm == 5.0
        assert lo.cover_summary_spacer_cm == 0.7
        assert lo.section_bottom_spacer_cm == 0.25
        assert lo.spacer_default_height_cm == 0.3
        assert lo.line_spacer_cm == 0.15
        assert lo.line_thickness == 0.6
        assert lo.image_max_width_cm == 15.0
        assert lo.image_max_height_cm == 9.0
        assert lo.plotly_export_width == 1200
        assert lo.plotly_export_height == 700
        assert lo.plotly_export_scale == 2
        assert lo.plotly_max_width_cm == 16.0
        assert lo.plotly_max_height_cm == 9.0
        assert lo.table_bottom_spacer_cm == 0.25
        assert lo.callout_width_cm == 16.2
        assert lo.callout_bottom_spacer_cm == 0.2

    def test_partial_override(self):
        lo = ReportLayout(spacer_default_height_cm=1.5, line_thickness=2.0)
        assert lo.spacer_default_height_cm == 1.5
        assert lo.line_thickness == 2.0
        assert lo.left_margin_cm == 2.0  # unchanged


# =============================================================================
# PDFReportEngine init
# =============================================================================

class TestPDFReportEngineInit:
    def test_defaults(self):
        engine = PDFReportEngine()
        assert isinstance(engine.theme, ReportTheme)
        assert isinstance(engine.layout, ReportLayout)

    def test_custom_theme_preserved(self):
        theme = ReportTheme(primary="#AABBCC")
        engine = PDFReportEngine(theme=theme)
        assert engine.theme.primary == "#AABBCC"

    def test_custom_layout_preserved(self):
        layout = ReportLayout(spacer_default_height_cm=2.0)
        engine = PDFReportEngine(layout=layout)
        assert engine.layout.spacer_default_height_cm == 2.0

    def test_all_renderer_keys_registered(self):
        engine = PDFReportEngine()
        expected = {
            "paragraph", "rich_text", "spacer", "line", "link", "image",
            "plotly", "table", "bullets", "numbered_list", "key_value",
            "callout", "code", "page_break", "subsection",
        }
        assert set(engine._renderers.keys()) == expected


# =============================================================================
# Block renderers — types and counts
# =============================================================================

class TestBlockRenderers:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = PDFReportEngine()

    def test_paragraph(self):
        result = self.engine._render_paragraph({"text": "Hello"})
        assert len(result) == 1 and isinstance(result[0], Paragraph)

    def test_rich_text(self):
        result = self.engine._render_rich_text({"text": "<b>Bold</b>"})
        assert len(result) == 1 and isinstance(result[0], Paragraph)

    def test_spacer_default_height(self):
        result = self.engine._render_spacer({})
        assert len(result) == 1
        assert isinstance(result[0], Spacer)
        expected = self.engine.layout.spacer_default_height_cm * cm
        assert abs(result[0].height - expected) < 1e-6

    def test_spacer_custom_height(self):
        result = self.engine._render_spacer({"height_cm": 2.0})
        assert isinstance(result[0], Spacer)
        assert abs(result[0].height - 2.0 * cm) < 1e-6

    def test_line_structure(self):
        result = self.engine._render_line({})
        assert len(result) == 3
        assert isinstance(result[0], Spacer)
        assert isinstance(result[1], HRFlowable)
        assert isinstance(result[2], Spacer)
        assert result[0].height == result[2].height

    def test_bullets(self):
        result = self.engine._render_bullets({"items": ["a", "b", "c"]})
        assert len(result) == 3
        assert all(isinstance(p, Paragraph) for p in result)

    def test_bullets_empty(self):
        assert self.engine._render_bullets({"items": []}) == []

    def test_numbered_list(self):
        result = self.engine._render_numbered_list({"items": ["first", "second"]})
        assert len(result) == 2
        assert all(isinstance(p, Paragraph) for p in result)

    def test_key_value(self):
        result = self.engine._render_key_value({
            "items": [{"key": "Name", "value": "Alice"}, {"key": "Age", "value": 30}]
        })
        assert len(result) == 2
        assert all(isinstance(p, Paragraph) for p in result)

    def test_page_break(self):
        result = self.engine._render_page_break({})
        assert len(result) == 1 and isinstance(result[0], PageBreak)

    def test_callout_structure(self):
        result = self.engine._render_callout({"text": "A note.", "title": "Note"})
        assert len(result) == 2
        assert isinstance(result[0], Table)
        assert isinstance(result[1], Spacer)

    def test_code_escapes_html(self):
        result = self.engine._render_code({"text": "x = 1 < 2 & True"})
        assert len(result) == 1 and isinstance(result[0], Paragraph)

    def test_unknown_block_type_returns_error_paragraph(self):
        result = self.engine._render_block({"type": "does_not_exist"})
        assert len(result) == 1 and isinstance(result[0], Paragraph)

    def test_image_missing_file_returns_error_paragraph(self):
        result = self.engine._render_image({"path": "/nonexistent/image.png"})
        assert len(result) == 1 and isinstance(result[0], Paragraph)

    def test_subsection(self):
        result = self.engine._render_subsection({
            "title": "Sub",
            "blocks": [{"type": "paragraph", "text": "Nested."}],
        })
        assert len(result) == 2
        assert isinstance(result[0], Paragraph)  # subtitle
        assert isinstance(result[1], Paragraph)  # body

    def test_table_wrapped_in_keep_together(self):
        result = self.engine._render_table({
            "headers": ["Col A", "Col B"],
            "rows": [["v1", "v2"], ["v3", "v4"]],
        })
        assert len(result) == 1 and isinstance(result[0], KeepTogether)

    def test_table_with_title_and_caption(self):
        result = self.engine._render_table({
            "title": "My Table",
            "headers": ["A"],
            "rows": [["1"]],
            "caption": "Table caption.",
        })
        kt = result[0]
        assert isinstance(kt, KeepTogether)


# =============================================================================
# Layout values are respected by renderers
# =============================================================================

class TestLayoutRespectedByRenderers:
    def test_spacer_default_height_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(spacer_default_height_cm=1.5))
        result = engine._render_spacer({})
        assert abs(result[0].height - 1.5 * cm) < 1e-6

    def test_line_spacer_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(line_spacer_cm=0.5))
        result = engine._render_line({})
        assert abs(result[0].height - 0.5 * cm) < 1e-6

    def test_line_thickness_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(line_thickness=2.0))
        result = engine._render_line({})
        assert result[1].lineWidth == 2.0

    def test_callout_spacer_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(callout_bottom_spacer_cm=0.8))
        result = engine._render_callout({"text": "Note.", "title": "Note"})
        assert abs(result[1].height - 0.8 * cm) < 1e-6

    def test_section_spacer_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(section_bottom_spacer_cm=1.0))
        result = engine._render_section({"title": "S", "blocks": []})
        # Last element of the section story should be the bottom spacer
        assert isinstance(result[-1], Spacer)
        assert abs(result[-1].height - 1.0 * cm) < 1e-6

    def test_table_spacer_from_layout(self):
        engine = PDFReportEngine(layout=ReportLayout(table_bottom_spacer_cm=0.9))
        # KeepTogether wraps the internal story — peek inside
        kt = engine._render_table({"headers": ["A"], "rows": [["1"]]})[0]
        internal = kt._content
        assert isinstance(internal[-1], Spacer)
        assert abs(internal[-1].height - 0.9 * cm) < 1e-6


# =============================================================================
# generate() — end-to-end
# =============================================================================

class TestGenerate:
    def test_creates_non_empty_pdf(self, minimal_report_data, tmp_path):
        output = str(tmp_path / "report.pdf")
        PDFReportEngine().generate(minimal_report_data, output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 1024  # at least 1 KB

    def test_missing_meta_raises(self):
        with pytest.raises(ValueError, match="'meta'"):
            PDFReportEngine().generate({"sections": []}, "/dev/null")

    def test_missing_title_raises(self):
        with pytest.raises(ValueError, match="'title'"):
            PDFReportEngine().generate({"meta": {}, "sections": []}, "/dev/null")

    def test_missing_sections_raises(self):
        with pytest.raises(ValueError, match="'sections'"):
            PDFReportEngine().generate({"meta": {"title": "T"}}, "/dev/null")

    def test_custom_layout_generates_pdf(self, minimal_report_data, tmp_path):
        output = str(tmp_path / "custom.pdf")
        engine = PDFReportEngine(layout=ReportLayout(
            cover_top_spacer_cm=2.0,
            line_thickness=1.5,
        ))
        engine.generate(minimal_report_data, output)
        assert Path(output).exists()

    def test_pdf_starts_with_pdf_magic_bytes(self, minimal_report_data, tmp_path):
        output = str(tmp_path / "magic.pdf")
        PDFReportEngine().generate(minimal_report_data, output)
        with open(output, "rb") as f:
            assert f.read(4) == b"%PDF"
