# -*- coding: utf-8 -*-
"""
Moteur PDF générique avec mini-format déclaratif `report_data`
Basé sur ReportLab.

Installation :
    pip install reportlab

Exécution :
    python generic_report_engine.py
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import utils
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

FONT_NAME = "LiberationSans"


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILE = "rapport_declaratif.pdf"


# ============================================================
# STYLES
# ============================================================

def register_fonts(font_dir: str):
    pdfmetrics.registerFont(TTFont("LiberationSans", f"{font_dir}/LiberationSans-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("LiberationSans-Bold", f"{font_dir}/LiberationSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("LiberationSans-Italic", f"{font_dir}/LiberationSans-Italic.ttf"))
    pdfmetrics.registerFont(TTFont("LiberationSans-BoldItalic", f"{font_dir}/LiberationSans-BoldItalic.ttf"))

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName=f"{FONT_NAME}-Bold",
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#16324F"),
        alignment=TA_CENTER,
        spaceAfter=16,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#5C6B7A"),
        alignment=TA_CENTER,
        spaceAfter=18,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontName=f"{FONT_NAME}-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#16324F"),
        spaceBefore=12,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SubSectionTitle",
        parent=styles["Heading2"],
        fontName=f"{FONT_NAME}-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2B4C6F"),
        spaceBefore=8,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#666666"),
        alignment=TA_LEFT,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Italic"],
        fontName=f"{FONT_NAME}-Italic",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
    ))

    styles.add(ParagraphStyle(
        name="Link",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#0B63CE"),
        alignment=TA_LEFT,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="CalloutTitle",
        parent=styles["BodyText"],
        fontName=f"{FONT_NAME}-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#16324F"),
        spaceAfter=3,
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


def build_table(data: List[List[Any]], col_widths=None, header_background="#16324F"):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_background)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D6E2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.whitesmoke,
            colors.HexColor("#F8FAFC"),
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
# DOC TEMPLATE
# ============================================================

def draw_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.setLineWidth(0.5)

    # header
    canvas.line(2 * cm, height - 1.8 * cm, width - 2 * cm, height - 1.8 * cm)
    canvas.setFont(f"{FONT_NAME}-Bold", 9)
    canvas.setFillColor(colors.HexColor("#16324F"))
    canvas.drawString(2 * cm, height - 1.4 * cm, doc.report_title)

    # footer
    canvas.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.0 * cm, f"Généré le {doc.generated_at}")
    canvas.drawRightString(width - 2 * cm, 1.0 * cm, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


class GenericReportDoc(BaseDocTemplate):
    def __init__(self, filename, report_title, generated_at, **kwargs):
        super().__init__(filename, **kwargs)
        self.report_title = report_title
        self.generated_at = generated_at

        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="main_frame"
        )

        template = PageTemplate(
            id="main_template",
            frames=[frame],
            onPage=draw_header_footer,
        )

        self.addPageTemplates([template])


# ============================================================
# RENDERER
# ============================================================

class PDFReportEngine:
    def __init__(self):
        self.styles = build_styles()
        self.block_renderers = {
            "paragraph": self.render_paragraph,
            "markdown_paragraph": self.render_markdown_paragraph,
            "spacer": self.render_spacer,
            "line": self.render_line,
            "link": self.render_link,
            "image": self.render_image,
            "table": self.render_table,
            "bullets": self.render_bullets,
            "key_value": self.render_key_value,
            "callout": self.render_callout,
            "page_break": self.render_page_break,
            "subsection": self.render_subsection,
        }

    # --------------------------------------------------------
    # API principale
    # --------------------------------------------------------

    def generate(self, report_data: Dict[str, Any], output_file: str):
        self.validate_report_data(report_data)

        doc = GenericReportDoc(
            output_file,
            report_title=report_data["meta"]["title"],
            generated_at=datetime.now().strftime("%d/%m/%Y à %H:%M"),
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

        doc.build(story)

    def validate_report_data(self, report_data: Dict[str, Any]):
        if "meta" not in report_data:
            raise ValueError("report_data doit contenir une clé 'meta'")
        if "title" not in report_data["meta"]:
            raise ValueError("report_data['meta'] doit contenir 'title'")
        if "sections" not in report_data:
            raise ValueError("report_data doit contenir une clé 'sections'")

    # --------------------------------------------------------
    # Structure haut niveau
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
        story = []
        story.append(Paragraph(section["title"], self.styles["SectionTitle"]))

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
                    self.styles["Body"]
                )
            ]
        return renderer(block)

    # --------------------------------------------------------
    # Rendu des blocs
    # --------------------------------------------------------

    def render_paragraph(self, block: Dict[str, Any]):
        return [Paragraph(block["text"], self.styles["Body"])]

    def render_markdown_paragraph(self, block: Dict[str, Any]):
        # Version simple : on suppose que le texte contient déjà
        # les balises compatibles ReportLab (<b>, <i>, <u>, <br/>...)
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
                color=colors.HexColor(block.get("color", "#D9E2EC")),
                spaceBefore=0,
                spaceAfter=0,
            ),
            Spacer(1, 0.15 * cm),
        ]

    def render_link(self, block: Dict[str, Any]):
        label = block.get("label", block["url"])
        url = block["url"]
        text = f'<link href="{url}" color="blue">{label}</link>'
        return [Paragraph(text, self.styles["Link"])]

    def render_image(self, block: Dict[str, Any]):
        story = []
        path = block["path"]

        if not file_exists(path):
            story.append(Paragraph(
                f"<font color='red'>[Image introuvable : {path}]</font>",
                self.styles["Body"]
            ))
            return story

        max_width_cm = block.get("max_width_cm", 15)
        max_height_cm = block.get("max_height_cm", 9)

        img = build_resized_image(
            path,
            max_width=max_width_cm * cm,
            max_height=max_height_cm * cm,
        )

        group = [img]

        caption = block.get("caption")
        if caption:
            group.append(Paragraph(caption, self.styles["Caption"]))

        story.append(KeepTogether(group))
        return story

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
            col_widths=col_widths,
            header_background=block.get("header_background", "#16324F"),
        )
        story.append(table)
        story.append(Spacer(1, 0.25 * cm))
        return [KeepTogether(story)]

    def render_bullets(self, block: Dict[str, Any]):
        story = []
        items = block.get("items", [])
        for item in items:
            story.append(Paragraph(f"• {safe_str(item)}", self.styles["Body"]))
        return story

    def render_key_value(self, block: Dict[str, Any]):
        story = []
        pairs = block.get("items", [])
        for pair in pairs:
            key = safe_str(pair.get("key"))
            value = safe_str(pair.get("value"))
            story.append(Paragraph(f"<b>{key} :</b> {value}", self.styles["Body"]))
        return story

    def render_callout(self, block: Dict[str, Any]):
        title = block.get("title", "Note")
        text = block["text"]

        content = [
            [Paragraph(f"<b>{title}</b>", self.styles["CalloutTitle"])],
            [Paragraph(text, self.styles["Body"])],
        ]

        table = Table(content, colWidths=[16.2 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(block.get("background", "#F4F7FB"))),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(block.get("border", "#B8C7D9"))),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return [table, Spacer(1, 0.2 * cm)]

    def render_page_break(self, block: Dict[str, Any]):
        return [PageBreak()]

    def render_subsection(self, block: Dict[str, Any]):
        story = [Paragraph(block["title"], self.styles["SubSectionTitle"])]
        for sub_block in block.get("blocks", []):
            story.extend(self.render_block(sub_block))
        return story


# ============================================================
# MINI-FORMAT DECLARATIF : EXEMPLE
# ============================================================

report_data = {
    "meta": {
        "title": "Rapport standardisé d'analyse",
        "subtitle": "Exemple avec mini-format déclaratif",
        "author": "Mon application",
        "date": datetime.now().strftime("%d/%m/%Y"),
        "version": "1.0.0",
        "summary": (
            "Ce rapport est généré à partir d'une structure déclarative. "
            "Le moteur PDF est générique et ne dépend pas du métier."
        ),
    },
    "sections": [
        {
            "title": "1. Introduction",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "L'objectif est de séparer complètement la production des données "
                        "du moteur de rendu PDF."
                    ),
                },
                {
                    "type": "paragraph",
                    "text": (
                        "Ainsi, ton application peut construire un objet <b>report_data</b> "
                        "sans connaître ReportLab en détail."
                    ),
                },
                {"type": "spacer", "height_cm": 0.2},
                {
                    "type": "link",
                    "label": "Documentation officielle ReportLab",
                    "url": "https://www.reportlab.com/documentation/",
                },
                {"type": "line"},
            ],
        },
        {
            "title": "2. Métadonnées et synthèse",
            "blocks": [
                {
                    "type": "key_value",
                    "items": [
                        {"key": "Projet", "value": "Démo PDF"},
                        {"key": "Environnement", "value": "Production"},
                        {"key": "Statut", "value": "Succès"},
                        {"key": "Nombre d'analyses", "value": 42},
                    ],
                },
                {
                    "type": "callout",
                    "title": "Info",
                    "text": (
                        "Le bloc <b>callout</b> permet d'afficher une note, une alerte "
                        "ou un point d'attention dans un encadré."
                    ),
                },
            ],
        },
        {
            "title": "3. Résultats détaillés",
            "blocks": [
                {
                    "type": "subsection",
                    "title": "3.1 Synthèse textuelle",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": (
                                "Les résultats montrent une performance globale stable, "
                                "avec une amélioration notable sur les cas standards."
                            ),
                        },
                        {
                            "type": "bullets",
                            "items": [
                                "Temps de traitement réduit",
                                "Stabilité correcte",
                                "Quelques cas limites à revoir",
                            ],
                        },
                    ],
                },
                {
                    "type": "subsection",
                    "title": "3.2 Tableau de résultats",
                    "blocks": [
                        {
                            "type": "table",
                            "title": "Tableau 1 - Résultats par échantillon",
                            "headers": ["ID", "Nom", "Statut", "Score"],
                            "rows": [
                                ["1", "Échantillon A", "OK", "95"],
                                ["2", "Échantillon B", "À revoir", "72"],
                                ["3", "Échantillon C", "OK", "88"],
                                ["4", "Échantillon D", "OK", "91"],
                            ],
                            "col_widths": [2, 6, 4, 3],  # en cm
                        }
                    ],
                },
            ],
        },
        {
            "title": "4. Illustration",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "Ce bloc montre comment intégrer une image déclarativement."
                    ),
                },
                {
                    "type": "image",
                    "path": "exemple_image.png",   # à remplacer
                    "caption": "Figure 1 - Exemple d'image dans le rapport",
                    "max_width_cm": 14,
                    "max_height_cm": 8,
                },
            ],
        },
        {
            "title": "5. Pagination et conclusion",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "Le moteur peut être enrichi avec d'autres types de blocs : "
                        "graphiques, annexes, sommaire, KPI, cartes, etc."
                    ),
                },
                {"type": "page_break"},
                {
                    "type": "subsection",
                    "title": "5.1 Nouvelle page",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": (
                                "Le bloc <b>page_break</b> permet de forcer un saut de page "
                                "dans la structure déclarative."
                            ),
                        }
                    ],
                },
            ],
        },
    ],
}


# ============================================================
# EXEMPLE : CONSTRUCTION DYNAMIQUE DE report_data
# ============================================================

def build_report_data_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exemple de fonction que ton application pourrait appeler.
    Elle convertit des données métier en mini-format déclaratif.
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
        ])

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
                "title": "2. Détail",
                "blocks": [
                    {
                        "type": "table",
                        "title": "Résultats détaillés",
                        "headers": ["ID", "Nom", "Statut", "Score"],
                        "rows": table_rows,
                    }
                ],
            },
        ],
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    register_fonts("/usr/share/fonts/truetype/liberation")
    engine = PDFReportEngine()
    engine.generate(report_data, OUTPUT_FILE)
    print(f"PDF généré : {OUTPUT_FILE}")