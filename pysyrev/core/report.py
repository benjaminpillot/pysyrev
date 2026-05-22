"""
Generic PDF report engine built on ReportLab.

Consumes a declarative `report_data` dict and renders it to a PDF file.
Domain-specific report builders (e.g. core/report_data.py) are responsible
for assembling that dict; this module only handles layout and rendering.

Supported block types
---------------------
paragraph, rich_text, spacer, line, link, image, plotly,
table, bullets, numbered_list, key_value, callout, code,
page_break, subsection

The `plotly` block type expects a pre-built ``plotly.graph_objects.Figure``
passed as ``block["figure"]``. plotly/kaleido must be installed separately.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors, utils
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# =============================================================================
# Theme & Layout
# =============================================================================

@dataclass
class ReportLayout:
    # Page margins
    left_margin_cm:   float = 2.0
    right_margin_cm:  float = 2.0
    top_margin_cm:    float = 2.4
    bottom_margin_cm: float = 2.2

    # Cover page
    cover_top_spacer_cm:     float = 5.0
    cover_summary_spacer_cm: float = 0.7

    # Section
    section_bottom_spacer_cm: float = 0.25

    # Spacer block default
    spacer_default_height_cm: float = 0.3

    # HR line
    line_spacer_cm:  float = 0.15
    line_thickness:  float = 0.6

    # Image defaults
    image_max_width_cm:  float = 15.0
    image_max_height_cm: float = 9.0

    # Plotly export defaults
    plotly_export_width:   int   = 1200
    plotly_export_height:  int   = 700
    plotly_export_scale:   int   = 2
    plotly_max_width_cm:   float = 16.0
    plotly_max_height_cm:  float = 9.0

    # Table
    table_bottom_spacer_cm: float = 0.25

    # Callout
    callout_width_cm:         float = 16.2
    callout_bottom_spacer_cm: float = 0.2


@dataclass
class ReportTheme:
    primary:          str = "#16324F"
    secondary:        str = "#2B4C6F"
    text:             str = "#000000"
    muted:            str = "#666666"
    line:             str = "#D9E2EC"
    table_header_bg:  str = "#16324F"
    table_grid:       str = "#C9D6E2"
    table_alt_row:    str = "#F8FAFC"
    callout_bg:       str = "#F4F7FB"
    callout_border:   str = "#B8C7D9"
    link:             str = "#0B63CE"


# =============================================================================
# Styles
# =============================================================================

def build_styles(theme: ReportTheme):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=colors.HexColor(theme.primary),
        alignment=TA_CENTER,
        spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#5C6B7A"),
        alignment=TA_CENTER,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor(theme.primary),
        spaceBefore=12,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SubSectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(theme.secondary),
        spaceBefore=8,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor(theme.text),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor(theme.muted),
        alignment=TA_LEFT,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor(theme.muted),
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="Link",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor(theme.link),
        alignment=TA_LEFT,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CalloutTitle",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor(theme.primary),
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ReportCode",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        backColor=colors.HexColor("#F6F8FA"),
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=6,
    ))

    return styles


# =============================================================================
# Helpers
# =============================================================================

def safe_str(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y %H:%M")
    return str(value)


def build_resized_image(path: str, max_width: float, max_height: float):
    img_reader = utils.ImageReader(path)
    width, height = img_reader.getSize()
    ratio = min(max_width / width, max_height / height)
    return Image(path, width=width * ratio, height=height * ratio)


def build_table(data: List[List[Any]], theme: ReportTheme,
                col_widths=None, header_background: Optional[str] = None,
                cell_style=None):
    # Wrap data-row strings in Paragraph so long words (e.g. DOIs) wrap properly.
    processed = []
    for i, row in enumerate(data):
        if i == 0 or cell_style is None:
            processed.append(list(row))
        else:
            processed.append([
                Paragraph(str(cell), cell_style) if isinstance(cell, str) else cell
                for cell in row
            ])
    table = Table(processed, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), colors.HexColor(header_background or theme.table_header_bg)),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("LEADING",      (0, 0), (-1, -1), 11),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor(theme.table_grid)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor(theme.table_alt_row)]),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    return table


# =============================================================================
# Plotly asset manager
# =============================================================================

class PlotlyAssetManager:
    """Exports Plotly figures to temporary PNG files for embedding in the PDF."""

    def __init__(self, base_dir: Optional[str] = None):
        self.temp_dir = Path(base_dir or tempfile.mkdtemp(prefix="report_assets_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def export_figure(self, fig, filename_prefix: str = "plot",
                      width: int = 1200, height: int = 700, scale: int = 2) -> str:
        self.counter += 1
        output_path = self.temp_dir / f"{filename_prefix}_{self.counter}.png"
        fig.write_image(str(output_path), width=width, height=height, scale=scale)
        return str(output_path)

    def cleanup(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# =============================================================================
# Document template
# =============================================================================

def _draw_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    canvas.setStrokeColor(colors.HexColor(doc.theme.line))
    canvas.setLineWidth(0.5)

    canvas.line(2 * cm, height - 1.8 * cm, width - 2 * cm, height - 1.8 * cm)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor(doc.theme.primary))
    canvas.drawString(2 * cm, height - 1.4 * cm, doc.report_title)

    canvas.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(doc.theme.muted))
    canvas.drawString(2 * cm, 1.0 * cm, f"Généré le {doc.generated_at}")
    canvas.drawRightString(width - 2 * cm, 1.0 * cm, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


class _ReportDoc(BaseDocTemplate):
    def __init__(self, filename, report_title, generated_at, theme: ReportTheme, **kwargs):
        super().__init__(filename, **kwargs)
        self.report_title = report_title
        self.generated_at = generated_at
        self.theme = theme

        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main_frame")
        self.addPageTemplates([
            PageTemplate(id="main_template", frames=[frame],
                         onPage=_draw_header_footer)
        ])


# =============================================================================
# PDF engine
# =============================================================================

class PDFReportEngine:
    """Render a declarative report_data dict to a PDF file.

    Parameters
    ----------
    theme : ReportTheme, optional
        Visual theme. Defaults to the built-in dark-blue palette.
    """

    def __init__(self, theme: Optional[ReportTheme] = None,
                 layout: Optional[ReportLayout] = None):
        self.theme = theme or ReportTheme()
        self.layout = layout or ReportLayout()
        self.styles = build_styles(self.theme)
        self.plotly_assets = PlotlyAssetManager()

        self._renderers = {
            "paragraph":     self._render_paragraph,
            "rich_text":     self._render_rich_text,
            "spacer":        self._render_spacer,
            "line":          self._render_line,
            "link":          self._render_link,
            "image":         self._render_image,
            "plotly":        self._render_plotly,
            "table":         self._render_table,
            "bullets":       self._render_bullets,
            "numbered_list": self._render_numbered_list,
            "key_value":     self._render_key_value,
            "callout":       self._render_callout,
            "code":          self._render_code,
            "page_break":    self._render_page_break,
            "subsection":    self._render_subsection,
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(self, report_data: Dict[str, Any], output_file: str):
        """Build the PDF from *report_data* and write it to *output_file*."""
        self._validate(report_data)

        lo = self.layout
        doc = _ReportDoc(
            output_file,
            report_title=report_data["meta"]["title"],
            generated_at=datetime.now().strftime("%d/%m/%Y à %H:%M"),
            theme=self.theme,
            pagesize=A4,
            leftMargin=lo.left_margin_cm * cm,
            rightMargin=lo.right_margin_cm * cm,
            topMargin=lo.top_margin_cm * cm,
            bottomMargin=lo.bottom_margin_cm * cm,
        )

        story = self._render_cover_page(report_data)
        for section in report_data.get("sections", []):
            story.extend(self._render_section(section))

        try:
            doc.build(story)
        finally:
            self.plotly_assets.cleanup()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate(report_data: Dict[str, Any]):
        if "meta" not in report_data:
            raise ValueError("report_data must contain a 'meta' key")
        if "title" not in report_data["meta"]:
            raise ValueError("report_data['meta'] must contain 'title'")
        if "sections" not in report_data:
            raise ValueError("report_data must contain a 'sections' key")

    # -------------------------------------------------------------------------
    # High-level renderers
    # -------------------------------------------------------------------------

    def _render_cover_page(self, report_data: Dict[str, Any]) -> list:
        lo = self.layout
        meta = report_data["meta"]
        story = [Spacer(1, lo.cover_top_spacer_cm * cm)]
        story.append(Paragraph(meta["title"], self.styles["ReportTitle"]))

        if meta.get("subtitle"):
            story.append(Paragraph(meta["subtitle"], self.styles["ReportSubtitle"]))

        for line in [
            f"<b>Auteur :</b> {safe_str(meta.get('author'))}",
            f"<b>Date :</b> {safe_str(meta.get('date'))}",
            f"<b>Version :</b> {safe_str(meta.get('version'))}",
        ]:
            story.append(Paragraph(line, self.styles["Body"]))

        if meta.get("summary"):
            story.append(Spacer(1, lo.cover_summary_spacer_cm * cm))
            story.append(Paragraph("Résumé", self.styles["SectionTitle"]))
            story.append(Paragraph(meta["summary"], self.styles["Body"]))

        story.append(PageBreak())
        return story

    def _render_section(self, section: Dict[str, Any]) -> list:
        story = [Paragraph(section["title"], self.styles["SectionTitle"])]
        for block in section.get("blocks", []):
            story.extend(self._render_block(block))
        story.append(Spacer(1, self.layout.section_bottom_spacer_cm * cm))
        return story

    def _render_block(self, block: Dict[str, Any]) -> list:
        renderer = self._renderers.get(block.get("type"))
        if not renderer:
            return [Paragraph(
                f"<font color='red'>Unsupported block type: {safe_str(block.get('type'))}</font>",
                self.styles["Body"],
            )]
        return renderer(block)

    # -------------------------------------------------------------------------
    # Block renderers
    # -------------------------------------------------------------------------

    def _render_paragraph(self, block):
        return [Paragraph(block["text"], self.styles["Body"])]

    def _render_rich_text(self, block):
        return [Paragraph(block["text"], self.styles["Body"])]

    def _render_spacer(self, block):
        return [Spacer(1, block.get("height_cm", self.layout.spacer_default_height_cm) * cm)]

    def _render_line(self, block):
        lo = self.layout
        spacer = Spacer(1, lo.line_spacer_cm * cm)
        return [
            spacer,
            HRFlowable(width="100%", thickness=lo.line_thickness,
                       color=colors.HexColor(block.get("color", self.theme.line)),
                       spaceBefore=0, spaceAfter=0),
            spacer,
        ]

    def _render_link(self, block):
        url = block["url"]
        label = block.get("label", url)
        if not any(url.startswith(s) for s in ("http://", "https://", "mailto:", "ftp://", "#")):
            url = Path(url).resolve().as_uri()
        text = f'<link href="{url}" color="{self.theme.link}">{label}</link>'
        return [Paragraph(text, self.styles["Link"])]

    def _render_image(self, block):
        lo = self.layout
        path = block["path"]
        if not Path(path).exists():
            return [Paragraph(
                f"<font color='red'>[Image not found: {path}]</font>",
                self.styles["Body"],
            )]
        img = build_resized_image(
            path,
            max_width=block.get("max_width_cm", lo.image_max_width_cm) * cm,
            max_height=block.get("max_height_cm", lo.image_max_height_cm) * cm,
        )
        group = [img]
        if block.get("caption"):
            group.append(Paragraph(block["caption"], self.styles["Caption"]))
        return [KeepTogether(group)]

    def _render_plotly(self, block):
        """Exports a plotly Figure to a temporary PNG, then delegates to _render_image."""
        lo = self.layout
        image_path = self.plotly_assets.export_figure(
            fig=block["figure"],
            filename_prefix=block.get("filename_prefix", "plot"),
            width=block.get("export_width", lo.plotly_export_width),
            height=block.get("export_height", lo.plotly_export_height),
            scale=block.get("export_scale", lo.plotly_export_scale),
        )
        return self._render_image({
            "type":          "image",
            "path":          image_path,
            "caption":       block.get("caption"),
            "max_width_cm":  block.get("max_width_cm", lo.plotly_max_width_cm),
            "max_height_cm": block.get("max_height_cm", lo.plotly_max_height_cm),
        })

    def _render_table(self, block):
        col_widths = [w * cm for w in block["col_widths"]] if block.get("col_widths") else None
        story = []
        if block.get("title"):
            story.append(Paragraph(block["title"], self.styles["SubSectionTitle"]))
        story.append(build_table(
            data=[block["headers"]] + block["rows"],
            theme=self.theme,
            col_widths=col_widths,
            header_background=block.get("header_background"),
            cell_style=self.styles["Small"],
        ))
        if block.get("caption"):
            story.append(Paragraph(block["caption"], self.styles["Caption"]))
        story.append(Spacer(1, self.layout.table_bottom_spacer_cm * cm))
        return [KeepTogether(story)]

    def _render_bullets(self, block):
        return [Paragraph(f"• {safe_str(item)}", self.styles["Body"])
                for item in block.get("items", [])]

    def _render_numbered_list(self, block):
        return [Paragraph(f"{i}. {safe_str(item)}", self.styles["Body"])
                for i, item in enumerate(block.get("items", []), start=1)]

    def _render_key_value(self, block):
        return [
            Paragraph(f"<b>{safe_str(p.get('key'))} :</b> {safe_str(p.get('value'))}",
                      self.styles["Body"])
            for p in block.get("items", [])
        ]

    def _render_callout(self, block):
        lo = self.layout
        width_cm = block.get("width_cm", lo.callout_width_cm)
        content = [
            [Paragraph(f"<b>{block.get('title', 'Note')}</b>", self.styles["CalloutTitle"])],
            [Paragraph(block["text"], self.styles["Body"])],
        ]
        table = Table(content, colWidths=[width_cm * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor(block.get("background", self.theme.callout_bg))),
            ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor(block.get("border", self.theme.callout_border))),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return [table, Spacer(1, lo.callout_bottom_spacer_cm * cm)]

    def _render_code(self, block):
        text = (block["text"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>"))
        return [Paragraph(text, self.styles["ReportCode"])]

    def _render_page_break(self, block):
        return [PageBreak()]

    def _render_subsection(self, block):
        story = [Paragraph(block["title"], self.styles["SubSectionTitle"])]
        for sub_block in block.get("blocks", []):
            story.extend(self._render_block(sub_block))
        return story
