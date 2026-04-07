# -*- coding: utf-8 -*-
"""
Exemple complet de moteur PDF générique + mini-format déclaratif
avec intégration d'images, tableaux, liens et graphiques Plotly.

Dépendances :
    pip install reportlab plotly kaleido

Exécution :
    python advanced_report_engine.py

Sortie :
    rapport_avance_plotly.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import tempfile
import shutil
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import utils

from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)

import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "rapport_avance_plotly.pdf"


# ============================================================
# THÈME
# ============================================================

@dataclass
class ReportTheme:
    primary: str = "#16324F"
    secondary: str = "#2B4C6F"
    text: str = "#000000"
    muted: str = "#666666"
    line: str = "#D9E2EC"
    table_header_bg: str = "#16324F"
    table_grid: str = "#C9D6E2"
    table_alt_row: str = "#F8FAFC"
    callout_bg: str = "#F4F7FB"
    callout_border: str = "#B8C7D9"
    link: str = "#0B63CE"


# ============================================================
# STYLES
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def safe_str(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y %H:%M")
    return str(value)


def file_exists(path: str) -> bool:
    return Path(path).exists()


def build_resized_image(path: str, max_width: float, max_height: float):
    img_reader = utils.ImageReader(path)
    width, height = img_reader.getSize()
    ratio = min(max_width / width, max_height / height)
    return Image(path, width=width * ratio, height=height * ratio)


def build_table(data: List[List[Any]], theme: ReportTheme, col_widths=None, header_background=None):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_background or theme.table_header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(theme.table_grid)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.whitesmoke,
            colors.HexColor(theme.table_alt_row),
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])
    table.setStyle(style)
    return table


# ============================================================
# PLOTLY -> IMAGE
# ============================================================

class PlotlyAssetManager:
    """
    Gère l'export des figures Plotly en PNG temporaires.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.temp_dir = Path(base_dir or tempfile.mkdtemp(prefix="report_assets_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def export_figure(self, fig, filename_prefix: str = "plot", width: int = 1200, height: int = 700, scale: int = 2) -> str:
        self.counter += 1
        output_path = self.temp_dir / f"{filename_prefix}_{self.counter}.png"
        fig.write_image(str(output_path), width=width, height=height, scale=scale)
        return str(output_path)

    def cleanup(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================
# DOC TEMPLATE
# ============================================================

def draw_header_footer(canvas, doc):
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


class GenericReportDoc(BaseDocTemplate):
    def __init__(self, filename, report_title, generated_at, theme: ReportTheme, **kwargs):
        super().__init__(filename, **kwargs)
        self.report_title = report_title
        self.generated_at = generated_at
        self.theme = theme

        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="main_frame",
        )

        template = PageTemplate(
            id="main_template",
            frames=[frame],
            onPage=draw_header_footer,
        )

        self.addPageTemplates([template])


# ============================================================
# MOTEUR PDF
# ============================================================

class PDFReportEngine:
    def __init__(self, theme: Optional[ReportTheme] = None):
        self.theme = theme or ReportTheme()
        self.styles = build_styles(self.theme)
        self.plotly_assets = PlotlyAssetManager()

        self.block_renderers = {
            "paragraph": self.render_paragraph,
            "rich_text": self.render_rich_text,
            "spacer": self.render_spacer,
            "line": self.render_line,
            "link": self.render_link,
            "image": self.render_image,
            "plotly": self.render_plotly,
            "table": self.render_table,
            "bullets": self.render_bullets,
            "numbered_list": self.render_numbered_list,
            "key_value": self.render_key_value,
            "callout": self.render_callout,
            "code": self.render_code,
            "page_break": self.render_page_break,
            "subsection": self.render_subsection,
        }

    # --------------------------------------------------------
    # API publique
    # --------------------------------------------------------

    def generate(self, report_data: Dict[str, Any], output_file: str):
        self.validate_report_data(report_data)

        doc = GenericReportDoc(
            output_file,
            report_title=report_data["meta"]["title"],
            generated_at=datetime.now().strftime("%d/%m/%Y à %H:%M"),
            theme=self.theme,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2.4 * cm,
            bottomMargin=2.2 * cm,
        )

        story = []
        story.extend(self.render_cover_page(report_data))

        for section in report_data.get("sections", []):
            story.extend(self.render_section(section))

        try:
            doc.build(story)
        finally:
            self.plotly_assets.cleanup()

    def validate_report_data(self, report_data: Dict[str, Any]):
        if "meta" not in report_data:
            raise ValueError("report_data doit contenir une clé 'meta'")
        if "title" not in report_data["meta"]:
            raise ValueError("report_data['meta'] doit contenir 'title'")
        if "sections" not in report_data:
            raise ValueError("report_data doit contenir une clé 'sections'")

    # --------------------------------------------------------
    # Rendu haut niveau
    # --------------------------------------------------------

    def render_cover_page(self, report_data: Dict[str, Any]):
        meta = report_data["meta"]
        story = []

        story.append(Spacer(1, 5 * cm))
        story.append(Paragraph(meta["title"], self.styles["ReportTitle"]))

        if meta.get("subtitle"):
            story.append(Paragraph(meta["subtitle"], self.styles["ReportSubtitle"]))

        info_lines = [
            f"<b>Auteur :</b> {safe_str(meta.get('author'))}",
            f"<b>Date :</b> {safe_str(meta.get('date'))}",
            f"<b>Version :</b> {safe_str(meta.get('version'))}",
        ]

        for line in info_lines:
            story.append(Paragraph(line, self.styles["Body"]))

        if meta.get("summary"):
            story.append(Spacer(1, 0.7 * cm))
            story.append(Paragraph("Résumé", self.styles["SectionTitle"]))
            story.append(Paragraph(meta["summary"], self.styles["Body"]))

        story.append(PageBreak())
        return story

    def render_section(self, section: Dict[str, Any]):
        story = [Paragraph(section["title"], self.styles["SectionTitle"])]
        for block in section.get("blocks", []):
            story.extend(self.render_block(block))
        story.append(Spacer(1, 0.25 * cm))
        return story

    def render_block(self, block: Dict[str, Any]):
        block_type = block.get("type")
        renderer = self.block_renderers.get(block_type)

        if not renderer:
            return [
                Paragraph(
                    f"<font color='red'>Bloc non supporté : {safe_str(block_type)}</font>",
                    self.styles["Body"],
                )
            ]
        return renderer(block)

    # --------------------------------------------------------
    # Renderers
    # --------------------------------------------------------

    def render_paragraph(self, block: Dict[str, Any]):
        return [Paragraph(block["text"], self.styles["Body"])]

    def render_rich_text(self, block: Dict[str, Any]):
        return [Paragraph(block["text"], self.styles["Body"])]

    def render_spacer(self, block: Dict[str, Any]):
        height_cm = block.get("height_cm", 0.3)
        return [Spacer(1, height_cm * cm)]

    def render_line(self, block: Dict[str, Any]):
        return [
            Spacer(1, 0.15 * cm),
            HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor(block.get("color", self.theme.line)),
                spaceBefore=0,
                spaceAfter=0,
            ),
            Spacer(1, 0.15 * cm),
        ]

    def render_link(self, block: Dict[str, Any]):
        label = block.get("label", block["url"])
        url = block["url"]
        text = f'<link href="{url}" color="{self.theme.link}">{label}</link>'
        return [Paragraph(text, self.styles["Link"])]

    def render_image(self, block: Dict[str, Any]):
        story = []
        path = block["path"]

        if not file_exists(path):
            return [Paragraph(
                f"<font color='red'>[Image introuvable : {path}]</font>",
                self.styles["Body"]
            )]

        max_width_cm = block.get("max_width_cm", 15)
        max_height_cm = block.get("max_height_cm", 9)

        img = build_resized_image(
            path,
            max_width=max_width_cm * cm,
            max_height=max_height_cm * cm,
        )

        group = [img]
        if block.get("caption"):
            group.append(Paragraph(block["caption"], self.styles["Caption"]))

        story.append(KeepTogether(group))
        return story

    def render_plotly(self, block: Dict[str, Any]):
        """
        Bloc attendu :
        {
            "type": "plotly",
            "figure": <plotly.graph_objects.Figure>,
            "caption": "...",
            "max_width_cm": 16,
            "max_height_cm": 9,
            "export_width": 1200,
            "export_height": 700,
            "export_scale": 2
        }
        """
        fig = block["figure"]

        image_path = self.plotly_assets.export_figure(
            fig=fig,
            filename_prefix=block.get("filename_prefix", "plot"),
            width=block.get("export_width", 1200),
            height=block.get("export_height", 700),
            scale=block.get("export_scale", 2),
        )

        image_block = {
            "type": "image",
            "path": image_path,
            "caption": block.get("caption"),
            "max_width_cm": block.get("max_width_cm", 16),
            "max_height_cm": block.get("max_height_cm", 9),
        }
        return self.render_image(image_block)

    def render_table(self, block: Dict[str, Any]):
        headers = block["headers"]
        rows = block["rows"]
        data = [headers] + rows

        col_widths = block.get("col_widths")
        if col_widths:
            col_widths = [w * cm for w in col_widths]

        story = []
        if block.get("title"):
            story.append(Paragraph(block["title"], self.styles["SubSectionTitle"]))

        table = build_table(
            data=data,
            theme=self.theme,
            col_widths=col_widths,
            header_background=block.get("header_background"),
        )
        story.append(table)

        if block.get("caption"):
            story.append(Paragraph(block["caption"], self.styles["Caption"]))

        story.append(Spacer(1, 0.25 * cm))
        return [KeepTogether(story)]

    def render_bullets(self, block: Dict[str, Any]):
        return [Paragraph(f"• {safe_str(item)}", self.styles["Body"]) for item in block.get("items", [])]

    def render_numbered_list(self, block: Dict[str, Any]):
        story = []
        for idx, item in enumerate(block.get("items", []), start=1):
            story.append(Paragraph(f"{idx}. {safe_str(item)}", self.styles["Body"]))
        return story

    def render_key_value(self, block: Dict[str, Any]):
        story = []
        for pair in block.get("items", []):
            key = safe_str(pair.get("key"))
            value = safe_str(pair.get("value"))
            story.append(Paragraph(f"<b>{key} :</b> {value}", self.styles["Body"]))
        return story

    def render_callout(self, block: Dict[str, Any]):
        title = block.get("title", "Note")
        text = block["text"]

        width_cm = block.get("width_cm", 16.2)

        content = [
            [Paragraph(f"<b>{title}</b>", self.styles["CalloutTitle"])],
            [Paragraph(text, self.styles["Body"])],
        ]

        table = Table(content, colWidths=[width_cm * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(block.get("background", self.theme.callout_bg))),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(block.get("border", self.theme.callout_border))),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return [table, Spacer(1, 0.2 * cm)]

    def render_code(self, block: Dict[str, Any]):
        text = block["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\n", "<br/>")
        return [Paragraph(text, self.styles["ReportCode"])]

    def render_page_break(self, block: Dict[str, Any]):
        return [PageBreak()]

    def render_subsection(self, block: Dict[str, Any]):
        story = [Paragraph(block["title"], self.styles["SubSectionTitle"])]
        for sub_block in block.get("blocks", []):
            story.extend(self.render_block(sub_block))
        return story


# ============================================================
# EXEMPLES DE FIGURES PLOTLY
# ============================================================

def create_bar_figure():
    categories = ["Jan", "Fév", "Mars", "Avr", "Mai", "Juin"]
    values = [120, 135, 150, 145, 170, 190]

    fig = px.bar(
        x=categories,
        y=values,
        labels={"x": "Mois", "y": "Volume"},
        title="Évolution mensuelle du volume traité",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def create_line_figure():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [72, 75, 74, 79, 83, 85, 84]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name="Score moyen",
    ))
    fig.update_layout(
        title="Progression du score moyen",
        xaxis_title="Itération",
        yaxis_title="Score",
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def create_pie_figure():
    fig = px.pie(
        names=["OK", "À revoir", "Erreur"],
        values=[78, 15, 7],
        title="Répartition des statuts",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


# ============================================================
# MINI-FORMAT DÉCLARATIF : EXEMPLE COMPLET
# ============================================================

def build_demo_report_data() -> Dict[str, Any]:
    bar_fig = create_bar_figure()
    line_fig = create_line_figure()
    pie_fig = create_pie_figure()

    return {
        "meta": {
            "title": "Bibliographic report - Pysyrev",
            "subtitle": "Exemple complet avec images, tableaux et plots Plotly",
            "author": "Report generated with the pysyrev engine",
            "date": datetime.now().strftime("%d/%m/%Y"),
            "version": "2.0.0",
            "summary": (
                "Ce document illustre un moteur PDF générique piloté par un mini-format "
                "déclaratif. Il prend en charge du texte, des liens, des images, des "
                "tableaux, des encadrés et des graphiques Plotly exportés en PNG."
            ),
        },
        "sections": [
            {
                "title": "1. Introduction",
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Le rapport est construit à partir d'une structure <b>report_data</b> "
                            "et non à partir d'appels ReportLab dispersés dans le code métier."
                        ),
                    },
                    {
                        "type": "rich_text",
                        "text": (
                            "Cette approche permet de <b>séparer le contenu</b> de la "
                            "<i>présentation</i>, ce qui simplifie fortement la maintenance."
                        ),
                    },
                    {
                        "type": "link",
                        "label": "Documentation ReportLab",
                        "url": "https://www.reportlab.com/documentation/",
                    },
                    {
                        "type": "link",
                        "label": "Documentation Plotly Python",
                        "url": "https://plotly.com/python/",
                    },
                    {"type": "line"},
                ],
            },
            {
                "title": "2. Métadonnées et éléments structurés",
                "blocks": [
                    {
                        "type": "key_value",
                        "items": [
                            {"key": "Projet", "value": "Génération PDF"},
                            {"key": "Pipeline", "value": "Analyse de batch"},
                            {"key": "Environnement", "value": "Production"},
                            {"key": "Statut global", "value": "Succès"},
                            {"key": "Nombre de lots", "value": 42},
                        ],
                    },
                    {
                        "type": "callout",
                        "title": "Point d'attention",
                        "text": (
                            "Pour intégrer Plotly dans un PDF, le plus robuste est "
                            "d'exporter chaque figure en image statique puis de l'insérer "
                            "comme n'importe quelle image."
                        ),
                    },
                    {
                        "type": "code",
                        "text": (
                            "fig.write_image('mon_plot.png', width=1200, height=700, scale=2)\n"
                            "# puis insertion du PNG dans ReportLab"
                        ),
                    },
                ],
            },
            {
                "title": "3. Résultats textuels et tabulaires",
                "blocks": [
                    {
                        "type": "subsection",
                        "title": "3.1 Synthèse",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "text": (
                                    "Les indicateurs montrent une amélioration progressive "
                                    "des performances sur les cas standards."
                                ),
                            },
                            {
                                "type": "bullets",
                                "items": [
                                    "Temps de traitement réduit",
                                    "Volume traité en hausse",
                                    "Taux d'erreur stabilisé",
                                    "Cas limites encore présents",
                                ],
                            },
                        ],
                    },
                    {
                        "type": "subsection",
                        "title": "3.2 Plan d'action",
                        "blocks": [
                            {
                                "type": "numbered_list",
                                "items": [
                                    "Vérifier les cas en anomalie",
                                    "Consolider les métriques hebdomadaires",
                                    "Industrialiser l'export automatique des rapports",
                                ],
                            }
                        ],
                    },
                    {
                        "type": "subsection",
                        "title": "3.3 Tableau détaillé",
                        "blocks": [
                            {
                                "type": "table",
                                "title": "Tableau 1 - Résultats par échantillon",
                                "headers": ["ID", "Nom", "Statut", "Score", "Durée (s)"],
                                "rows": [
                                    ["1", "Échantillon A", "OK", "95", "1.4"],
                                    ["2", "Échantillon B", "À revoir", "72", "1.9"],
                                    ["3", "Échantillon C", "OK", "88", "1.5"],
                                    ["4", "Échantillon D", "OK", "91", "1.6"],
                                    ["5", "Échantillon E", "Erreur", "40", "2.3"],
                                ],
                                "col_widths": [1.5, 5.0, 3.5, 2.5, 3.0],
                                "caption": "Exemple de tableau rendu par le moteur générique.",
                            }
                        ],
                    },
                ],
            },
            {
                "title": "4. Illustrations et médias",
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Le moteur sait aussi intégrer des images statiques déjà présentes "
                            "sur disque."
                        ),
                    },
                    {
                        "type": "image",
                        "path": "exemple_image.png",  # à remplacer par un fichier réel
                        "caption": "Figure 1 - Exemple d'image locale",
                        "max_width_cm": 14,
                        "max_height_cm": 8,
                    },
                ],
            },
            {
                "title": "5. Visualisations Plotly",
                "blocks": [
                    {
                        "type": "subsection",
                        "title": "5.1 Histogramme",
                        "blocks": [
                            {
                                "type": "plotly",
                                "figure": bar_fig,
                                "caption": "Figure 2 - Volume traité par mois",
                                "max_width_cm": 16,
                                "max_height_cm": 8.5,
                            }
                        ],
                    },
                    {
                        "type": "subsection",
                        "title": "5.2 Courbe de progression",
                        "blocks": [
                            {
                                "type": "plotly",
                                "figure": line_fig,
                                "caption": "Figure 3 - Progression du score moyen",
                                "max_width_cm": 16,
                                "max_height_cm": 8.5,
                            }
                        ],
                    },
                    {
                        "type": "subsection",
                        "title": "5.3 Répartition",
                        "blocks": [
                            {
                                "type": "plotly",
                                "figure": pie_fig,
                                "caption": "Figure 4 - Répartition des statuts",
                                "max_width_cm": 13,
                                "max_height_cm": 8.5,
                            }
                        ],
                    },
                ],
            },
            {
                "title": "6. Pagination et conclusion",
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Le mini-format peut être enrichi facilement avec de nouveaux blocs "
                            "comme des KPI, des graphiques Matplotlib, des annexes ou un sommaire."
                        ),
                    },
                    {"type": "page_break"},
                    {
                        "type": "subsection",
                        "title": "6.1 Nouvelle page",
                        "blocks": [
                            {
                                "type": "callout",
                                "title": "Conclusion",
                                "text": (
                                    "Avec cette architecture, ton application peut simplement "
                                    "produire un objet déclaratif, puis laisser le moteur PDF "
                                    "gérer tout le rendu."
                                ),
                            }
                        ],
                    },
                ],
            },
        ],
    }


# ============================================================
# EXEMPLE DE CONVERSION DONNÉES MÉTIER -> report_data
# ============================================================

def build_report_data_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exemple de transformation de données métier vers le mini-format.
    """

    metric_pairs = [
        {"key": key, "value": value}
        for key, value in results.get("metrics", {}).items()
    ]

    table_rows = []
    for item in results.get("samples", []):
        table_rows.append([
            safe_str(item.get("id")),
            safe_str(item.get("name")),
            safe_str(item.get("status")),
            safe_str(item.get("score")),
            safe_str(item.get("duration")),
        ])

    # Exemple de figure dynamique
    fig = px.bar(
        x=[sample["name"] for sample in results.get("samples", [])],
        y=[sample["score"] for sample in results.get("samples", [])],
        labels={"x": "Échantillon", "y": "Score"},
        title="Score par échantillon",
    )
    fig.update_layout(template="plotly_white")

    return {
        "meta": {
            "title": results.get("title", "Rapport d'exécution"),
            "subtitle": results.get("subtitle", "Synthèse automatique"),
            "author": results.get("author", "Application"),
            "date": datetime.now().strftime("%d/%m/%Y"),
            "version": results.get("version", "1.0"),
            "summary": results.get("summary", ""),
        },
        "sections": [
            {
                "title": "1. Résumé",
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": results.get("intro", "Aucun texte d'introduction."),
                    },
                    {
                        "type": "key_value",
                        "items": metric_pairs,
                    },
                ],
            },
            {
                "title": "2. Résultats détaillés",
                "blocks": [
                    {
                        "type": "table",
                        "title": "Résultats détaillés",
                        "headers": ["ID", "Nom", "Statut", "Score", "Durée"],
                        "rows": table_rows,
                    },
                    {
                        "type": "plotly",
                        "figure": fig,
                        "caption": "Visualisation dynamique issue des données métier",
                    },
                ],
            },
        ],
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Démo autonome
    report_data = build_demo_report_data()

    engine = PDFReportEngine()
    engine.generate(report_data, OUTPUT_FILE)

    print(f"PDF généré : {OUTPUT_FILE}")