"""
HTML report engine — the interactive twin of the PDF one.

Consumes the same declarative ``report_data`` dict as
:class:`~pysyrev.core.report.engine.PDFReportEngine` and writes one
self-contained HTML page: a sidebar listing the sections, one section shown at
a time, the Plotly figures kept live (zoom, hover, legend toggling) instead of
being flattened to PNG, and the tables sortable and filterable.

The page carries no dependency on the run directory — figures, plotly.js and
images are all embedded — so the file can be moved or sent as is.

A block may carry ``"only": "pdf"`` (or ``"html"``) to be rendered by one
engine only: the "open the interactive graph" pointers the PDF needs are noise
in a page whose graphs are already interactive.
"""

from __future__ import annotations

import base64
import html as _html
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pysyrev.core.report.engine import ReportTheme, validate_report_data


# =============================================================================
# Text
# =============================================================================
#
# Section builders write their texts in ReportLab's inline markup (<b>, <br/>,
# <link href=…>, <font color=…>), and paper titles or LLM labels reach them
# unescaped. So the text is escaped whole, and only that small markup is then
# given back its meaning — anything else stays the literal text it was.

_Q = r"(?:&quot;|&#x27;)"
_INLINE_TAG = re.compile(r"&lt;(/?)(b|i|u|em|strong|sub|sup)&gt;", re.I)
_BREAK_TAG = re.compile(r"&lt;br\s*/?&gt;", re.I)
_LINK_TAG = re.compile(
    rf"&lt;(?:link|a)\s+href={_Q}(.*?){_Q}.*?&gt;(.*?)&lt;/(?:link|a)&gt;", re.I | re.S)
_FONT_TAG = re.compile(
    rf"&lt;font\s+color={_Q}([#\w]+){_Q}\s*&gt;(.*?)&lt;/font&gt;", re.I | re.S)
_SAFE_HREF = re.compile(r"^(https?://|mailto:|ftp://|file://|#)", re.I)

_URL_CELL = re.compile(r"^https?://\S+$", re.I)
_DOI_CELL = re.compile(r"^10\.\d{4,9}/\S+$")


def _link(match: re.Match) -> str:
    href, label = match.group(1), match.group(2)
    if not _SAFE_HREF.match(href):
        return label
    return f'<a href="{href}" target="_blank" rel="noopener">{label}</a>'


def html_text(value: Any) -> str:
    """*value* as HTML: escaped, with the report's inline markup restored."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        value = value.strftime("%d/%m/%Y %H:%M")
    text = _html.escape(str(value), quote=True)
    text = _INLINE_TAG.sub(lambda m: f"<{m.group(1)}{m.group(2).lower()}>", text)
    text = _BREAK_TAG.sub("<br>", text)
    text = _LINK_TAG.sub(_link, text)
    text = _FONT_TAG.sub(r'<span style="color:\1">\2</span>', text)
    return text


def _cell(value: Any) -> str:
    """A table cell: a bare URL or DOI becomes a link, the rest is text."""
    raw = "" if value is None else str(value).strip()
    if _URL_CELL.match(raw):
        href = _html.escape(raw, quote=True)
        return f'<a href="{href}" target="_blank" rel="noopener">{href}</a>'
    if _DOI_CELL.match(raw):
        doi = _html.escape(raw, quote=True)
        return f'<a href="https://doi.org/{doi}" target="_blank" rel="noopener">{doi}</a>'
    return html_text(value)


def _script_json(payload: str) -> str:
    """*payload* (JSON) made safe to sit inside a ``<script>`` element.

    ``<`` only occurs inside JSON strings, where ``\\u003c`` means the same —
    and without it a ``</script>`` in a paper title would end the element.
    """
    return payload.replace("<", "\\u003c")


# =============================================================================
# Page assets
# =============================================================================

_CSS = """
:root {
  --primary: %(primary)s; --secondary: %(secondary)s; --text: #1b2733;
  --muted: %(muted)s; --line: %(line)s; --grid: %(table_grid)s;
  --alt-row: %(table_alt_row)s; --callout-bg: %(callout_bg)s;
  --callout-border: %(callout_border)s; --link: %(link)s;
  --sidebar-w: 290px;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%%; }
body {
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--text); background: #fff;
}
a { color: var(--link); }
.layout { display: flex; min-height: 100%%; }

/* ---- sidebar --------------------------------------------------------- */
.sidebar {
  flex: 0 0 var(--sidebar-w); width: var(--sidebar-w);
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  background: var(--primary); color: #fff;
  display: flex; flex-direction: column; padding: 22px 0 14px;
}
.sidebar h1 { font-size: 18px; line-height: 1.3; margin: 0 20px 6px; }
.sidebar .subtitle { margin: 0 20px 16px; font-size: 13px; opacity: .8; }
.sidebar nav { flex: 1; }
.sidebar nav a {
  display: block; padding: 8px 20px; color: #fff; text-decoration: none;
  font-size: 14px; border-left: 3px solid transparent; opacity: .82;
}
.sidebar nav a:hover { opacity: 1; background: rgba(255,255,255,.08); }
.sidebar nav a.active {
  opacity: 1; font-weight: 600; border-left-color: #fff;
  background: rgba(255,255,255,.14);
}
.sidebar nav .subnav { display: none; }
.sidebar nav .subnav.open { display: block; }
.sidebar nav .subnav a { padding: 4px 20px 4px 36px; font-size: 13px; opacity: .7; }
.sidebar .meta {
  margin: 16px 20px 0; padding-top: 12px; font-size: 12px; opacity: .75;
  border-top: 1px solid rgba(255,255,255,.25);
}
.sidebar .meta div { margin: 2px 0; }

/* ---- content --------------------------------------------------------- */
main { flex: 1; min-width: 0; padding: 28px 40px 60px; }
.section { display: none; max-width: 1250px; margin: 0 auto; }
.section.active { display: block; }
.section > h2 {
  font-size: 24px; color: var(--primary); margin: 0 0 18px;
  padding-bottom: 10px; border-bottom: 2px solid var(--line);
}
h3 { font-size: 18px; color: var(--secondary); margin: 30px 0 10px; scroll-margin-top: 16px; }
h4 { font-size: 15px; color: var(--secondary); margin: 22px 0 8px; }
p { margin: 0 0 12px; }
hr { border: 0; border-top: 1px solid var(--line); margin: 14px 0; }
ul, ol { margin: 0 0 12px; padding-left: 24px; }
.kv { margin: 0 0 14px; }
.kv div { margin: 2px 0; }
.callout {
  background: var(--callout-bg); border: 1px solid var(--callout-border);
  border-radius: 6px; padding: 10px 14px; margin: 0 0 14px;
}
.callout .callout-title { font-weight: 700; color: var(--primary); margin-bottom: 4px; }
pre {
  background: #f5f7fa; border: 1px solid var(--line); border-radius: 6px;
  padding: 10px 12px; overflow-x: auto; font-size: 13px;
}
.caption { color: var(--muted); font-size: 13px; font-style: italic; margin: 6px 0 16px; }
.error { color: #c0392b; }
.spacer { height: 10px; }
figure { margin: 0 0 8px; }
figure img { max-width: 100%%; height: auto; }
.figure-plot { width: 100%%; border: 1px solid var(--line); border-radius: 6px; }

/* ---- tables ---------------------------------------------------------- */
.table-block { margin: 0 0 18px; }
.table-tools { display: flex; align-items: center; gap: 12px; margin: 0 0 6px; }
.table-tools input {
  font: inherit; font-size: 13px; padding: 5px 9px; width: 260px;
  border: 1px solid var(--grid); border-radius: 5px;
}
.table-tools .count { font-size: 12px; color: var(--muted); }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%%; font-size: 13px; }
th, td { border: 1px solid var(--grid); padding: 6px 9px; text-align: left; vertical-align: top; }
th {
  background: var(--primary); color: #fff; font-weight: 600;
  cursor: pointer; user-select: none; white-space: nowrap;
}
th::after { content: " \\2195"; opacity: .45; font-size: 11px; }
th[data-dir="asc"]::after { content: " \\2191"; opacity: 1; }
th[data-dir="desc"]::after { content: " \\2193"; opacity: 1; }
tbody tr:nth-child(even) { background: var(--alt-row); }
tbody tr:hover { background: #e9f0f8; }
td { overflow-wrap: anywhere; }

@media (max-width: 900px) {
  .layout { display: block; }
  .sidebar { position: static; width: auto; height: auto; }
  main { padding: 20px 16px 40px; }
}
"""

_JS = """
(function () {
  var sections = Array.prototype.slice.call(document.querySelectorAll('.section'));
  var links = Array.prototype.slice.call(document.querySelectorAll('.sidebar nav a[data-section]'));

  // Figures are drawn the first time their section is shown: a hidden
  // container has no size to draw into, and a page of twenty networks would
  // otherwise all be laid out before the reader sees the first one.
  function renderFigures(root) {
    if (!window.Plotly) { return; }
    root.querySelectorAll('.figure-plot:not([data-done])').forEach(function (el) {
      var spec = JSON.parse(el.previousElementSibling.textContent);
      window.Plotly.newPlot(el, spec.data, spec.layout, {responsive: true, displaylogo: false});
      el.setAttribute('data-done', '1');
    });
  }

  function show(id, anchor) {
    var target = document.getElementById(id) || sections[0];
    if (!target) { return; }
    sections.forEach(function (s) { s.classList.toggle('active', s === target); });
    links.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('data-section') === target.id);
    });
    document.querySelectorAll('.subnav').forEach(function (n) {
      n.classList.toggle('open', n.getAttribute('data-for') === target.id);
    });
    renderFigures(target);
    var el = anchor && document.getElementById(anchor);
    if (el) { el.scrollIntoView(); } else { window.scrollTo(0, 0); }
    document.dispatchEvent(new CustomEvent('report:section', {detail: target}));
  }

  function route() {
    var hash = decodeURIComponent(location.hash.slice(1));
    var el = hash && document.getElementById(hash);
    var section = el && el.closest('.section');
    if (section) { show(section.id, section === el ? null : hash); }
    else { show(sections.length ? sections[0].id : null); }
  }

  // ---- tables: click a header to sort, type to filter ----
  function sortKey(text) {
    var t = text.trim().replace(/[\\s\\u00a0]/g, '');
    if (/^-?[\\d.,]+%?$/.test(t)) {
      var n = parseFloat(t.replace(/,/g, '').replace('%', ''));
      if (!isNaN(n)) { return n; }
    }
    return null;
  }
  function bindTable(block) {
    var tbody = block.querySelector('tbody');
    var heads = block.querySelectorAll('th');
    heads.forEach(function (th, col) {
      th.addEventListener('click', function () {
        var dir = th.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';
        heads.forEach(function (h) { h.removeAttribute('data-dir'); });
        th.setAttribute('data-dir', dir);
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) {
          var x = a.cells[col].textContent, y = b.cells[col].textContent;
          var nx = sortKey(x), ny = sortKey(y);
          var cmp = (nx !== null && ny !== null) ? nx - ny
                  : x.localeCompare(y, undefined, {numeric: true, sensitivity: 'base'});
          return dir === 'asc' ? cmp : -cmp;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
    var input = block.querySelector('.table-tools input');
    if (!input) { return; }
    var count = block.querySelector('.count');
    input.addEventListener('input', function () {
      var q = input.value.toLowerCase(), shown = 0;
      Array.prototype.forEach.call(tbody.rows, function (r) {
        var hit = !q || r.textContent.toLowerCase().indexOf(q) !== -1;
        r.style.display = hit ? '' : 'none';
        if (hit) { shown += 1; }
      });
      count.textContent = shown + ' / ' + tbody.rows.length;
    });
  }
  function bindTables(root) {
    root.querySelectorAll('.table-block').forEach(bindTable);
  }

  // What a script that replaces a section's content needs to bring it to life.
  window.report = {renderFigures: renderFigures, bindTables: bindTables};

  bindTables(document);
  window.addEventListener('hashchange', route);
  route();
})();
"""

# Tables shorter than this are read at a glance: no filter box.
_FILTER_MIN_ROWS = 10
_DEFAULT_FIGURE_HEIGHT = 560


# =============================================================================
# HTML engine
# =============================================================================

class HTMLReportEngine:
    """Render a declarative report_data dict to one self-contained HTML page.

    Parameters
    ----------
    theme : ReportTheme, optional
        The PDF engine's theme — the page takes its colours from it.
    plotlyjs : {"embed", "cdn"}
        ``embed`` (default) inlines plotly.js: a few MB heavier, but the file
        works offline and on its own. ``cdn`` loads it from the network.
    """

    def __init__(self, theme: Optional[ReportTheme] = None, plotlyjs: str = "embed"):
        if plotlyjs not in ("embed", "cdn"):
            raise ValueError(f"plotlyjs must be 'embed' or 'cdn', got {plotlyjs!r}")
        self.theme = theme or ReportTheme()
        self.plotlyjs = plotlyjs
        self._has_figures = False
        self._section_id = ""
        self._subsections: List[tuple] = []

        self._renderers = {
            "paragraph":     self._render_paragraph,
            "rich_text":     self._render_paragraph,
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

    def generate(self, report_data: Dict[str, Any], output_file: str) -> None:
        """Build the page from *report_data* and write it to *output_file*."""
        Path(output_file).write_text(self.render(report_data), encoding="utf-8")

    def render(self, report_data: Dict[str, Any], *, extra_css: str = "",
               extra_html: str = "", extra_js: str = "",
               force_plotlyjs: bool = False) -> str:
        """The whole page, as a string.

        The *extra_* arguments are how a caller grafts its own panel onto the
        page (the live report's settings): CSS appended to the page's, HTML
        placed beside the content, a script run after the page's own.
        *force_plotlyjs* loads plotly.js even when no section draws a figure
        yet.
        """
        validate_report_data(report_data)
        self._has_figures = False
        meta = report_data["meta"]

        panels, nav = [], []
        if meta.get("summary"):
            panels.append(self._panel("summary", None, "<h2>Résumé</h2>"
                                      f"<p>{html_text(meta['summary'])}</p>"))
            nav.append(self._nav_entry("summary", "Résumé", ""))
        for sid, section in self.section_ids(report_data):
            body, subnav = self.render_section(section, sid)
            panels.append(self._panel(sid, section.get("key"), body))
            nav.append(self._nav_entry(sid, section["title"], subnav))

        title = html_text(meta["title"])
        subtitle = (f'<div class="subtitle">{html_text(meta["subtitle"])}</div>'
                    if meta.get("subtitle") else "")
        meta_lines = "".join(
            f"<div><b>{label} :</b> {html_text(meta[key])}</div>"
            for label, key in (("Auteur", "author"), ("Date", "date"),
                               ("Version", "version"))
            if meta.get(key))
        css = _CSS % self.theme.__dict__ + extra_css
        plotly = (self._plotly_script()
                  if self._has_figures or force_plotlyjs else "")
        extra_script = f"<script>{extra_js}</script>\n" if extra_js else ""

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{_html.escape(str(meta['title']))}</title>\n"
            f"<style>{css}</style>\n</head>\n<body>\n"
            '<div class="layout">\n'
            f'<aside class="sidebar"><h1>{title}</h1>{subtitle}'
            f'<nav>{"".join(nav)}</nav>'
            f'<div class="meta">{meta_lines}</div></aside>\n'
            f'<main>{"".join(panels)}</main>\n'
            f"{extra_html}"
            "</div>\n"
            f"{plotly}\n"
            f"<script>{_JS}</script>\n"
            f"{extra_script}"
            "</body>\n</html>\n"
        )

    @staticmethod
    def section_ids(report_data: Dict[str, Any]) -> List[tuple]:
        """``(element id, section)`` for each section, in page order."""
        return [(f"section-{i}", section)
                for i, section in enumerate(report_data.get("sections", []), start=1)]

    def render_section(self, section: Dict[str, Any], sid: str) -> tuple:
        """One section as ``(content, subnav)`` HTML fragments.

        *content* is what goes inside the section's element — heading, blocks
        and the data of its figures — and *subnav* the sidebar links to its
        subsections. Self-contained, so a section can be swapped in a page
        already open.
        """
        self._section_id = sid
        self._subsections: List[tuple] = []
        body = "".join(self._render_block(b, depth=0)
                       for b in section.get("blocks", []))
        subnav = "".join(f'<a href="#{aid}">{html_text(t)}</a>'
                         for aid, t in self._subsections)
        return f"<h2>{html_text(section['title'])}</h2>{body}", subnav

    # -------------------------------------------------------------------------
    # Page structure
    # -------------------------------------------------------------------------

    @staticmethod
    def _panel(sid: str, key: Optional[str], content: str) -> str:
        data_key = f' data-key="{_html.escape(str(key), quote=True)}"' if key else ""
        return f'<section class="section" id="{sid}"{data_key}>{content}</section>\n'

    @staticmethod
    def _nav_entry(sid: str, title: Any, subnav: str) -> str:
        return (f'<a href="#{sid}" data-section="{sid}">{html_text(title)}</a>'
                f'<div class="subnav" data-for="{sid}">{subnav}</div>')

    def _plotly_script(self) -> str:
        from plotly.offline import get_plotlyjs, get_plotlyjs_version
        if self.plotlyjs == "cdn":
            return (f'<script src="https://cdn.plot.ly/plotly-'
                    f'{get_plotlyjs_version()}.min.js" charset="utf-8"></script>')
        return f"<script>{get_plotlyjs()}</script>"

    # -------------------------------------------------------------------------
    # Block renderers
    # -------------------------------------------------------------------------

    def _render_block(self, block: Dict[str, Any], depth: int) -> str:
        if block.get("only") not in (None, "html"):
            return ""
        renderer = self._renderers.get(block.get("type"))
        if not renderer:
            return (f'<p class="error">Unsupported block type: '
                    f'{html_text(block.get("type"))}</p>')
        if renderer == self._render_subsection:
            return self._render_subsection(block, depth)
        return renderer(block)

    def _render_paragraph(self, block):
        return f"<p>{html_text(block['text'])}</p>"

    def _render_spacer(self, block):
        return '<div class="spacer"></div>'

    def _render_line(self, block):
        return "<hr>"

    def _render_link(self, block):
        url = str(block["url"])
        label = block.get("label", url)
        if not _SAFE_HREF.match(url):
            url = Path(url).resolve().as_uri()
        return (f'<p><a href="{_html.escape(url, quote=True)}" target="_blank" '
                f'rel="noopener">{html_text(label)}</a></p>')

    def _render_image(self, block):
        path = Path(block["path"])
        if not path.exists():
            return f'<p class="error">[Image not found: {html_text(path)}]</p>'
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return (f'<figure><img alt="" src="data:{mime};base64,{data}">'
                f"{self._caption(block)}</figure>")

    def _render_plotly(self, block):
        import plotly.graph_objects as go

        # The figure is sized for its PNG in the PDF; here it fills the column
        # and keeps its height.
        fig = go.Figure(block["figure"])
        fig.update_layout(autosize=True, width=None,
                          height=fig.layout.height or _DEFAULT_FIGURE_HEIGHT)
        self._has_figures = True
        # The data sits beside its figure, so the pair travels as one fragment.
        return ('<figure><script type="application/json" class="figure-spec">'
                f"{_script_json(fig.to_json())}</script>"
                f'<div class="figure-plot" style="height:{int(fig.layout.height)}px"></div>'
                f"{self._caption(block)}</figure>")

    def _render_table(self, block):
        headers, rows = block["headers"], block["rows"]
        out = ['<div class="table-block">']
        if block.get("title"):
            out.append(f"<h4>{html_text(block['title'])}</h4>")
        if len(rows) >= _FILTER_MIN_ROWS:
            out.append('<div class="table-tools">'
                       '<input type="search" placeholder="Filter rows…">'
                       f'<span class="count">{len(rows)} / {len(rows)}</span></div>')
        out.append('<div class="table-scroll"><table>')
        widths = block.get("col_widths")
        if widths and len(widths) == len(headers) and sum(widths) > 0:
            total = float(sum(widths))
            out.append("<colgroup>" + "".join(
                f'<col style="width:{100 * w / total:.1f}%">' for w in widths)
                + "</colgroup>")
        out.append("<thead><tr>"
                   + "".join(f"<th>{html_text(h)}</th>" for h in headers)
                   + "</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table></div>")
        out.append(self._caption(block))
        out.append("</div>")
        return "".join(out)

    def _render_bullets(self, block):
        return "<ul>" + "".join(f"<li>{html_text(i)}</li>"
                                for i in block.get("items", [])) + "</ul>"

    def _render_numbered_list(self, block):
        return "<ol>" + "".join(f"<li>{html_text(i)}</li>"
                                for i in block.get("items", [])) + "</ol>"

    def _render_key_value(self, block):
        return '<div class="kv">' + "".join(
            f"<div><b>{html_text(p.get('key'))} :</b> {html_text(p.get('value'))}</div>"
            for p in block.get("items", [])) + "</div>"

    def _render_callout(self, block):
        return ('<div class="callout">'
                f'<div class="callout-title">{html_text(block.get("title", "Note"))}</div>'
                f"<div>{html_text(block['text'])}</div></div>")

    def _render_code(self, block):
        return f"<pre><code>{_html.escape(str(block['text']))}</code></pre>"

    def _render_page_break(self, block):
        return ""

    def _render_subsection(self, block, depth: int = 0):
        # Only a section's own subsections are listed in the sidebar; the ones
        # nested below them are headings within the page.
        if depth == 0:
            anchor = f"{self._section_id}-{len(self._subsections) + 1}"
            self._subsections.append((anchor, block["title"]))
            head = f'<h3 id="{anchor}">{html_text(block["title"])}</h3>'
        else:
            head = f"<h4>{html_text(block['title'])}</h4>"
        return head + "".join(self._render_block(b, depth + 1)
                              for b in block.get("blocks", []))

    @staticmethod
    def _caption(block) -> str:
        if not block.get("caption"):
            return ""
        return f'<div class="caption">{html_text(block["caption"])}</div>'
