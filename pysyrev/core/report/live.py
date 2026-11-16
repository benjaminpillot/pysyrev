"""
Live report — the HTML report served locally, with settings one can turn.

The page is the one :class:`~pysyrev.core.report.html.HTMLReportEngine` writes,
plus a settings panel. Changing a setting rebuilds *one section* from the data
already in memory (see ``rebuilders`` in
:func:`~pysyrev.core.report.data.build_report_data`) and swaps it in the open
page; nothing is written to disk. The panel also shows the changed settings as
the YAML to paste back under ``topic_report.sections``.

Only settings that are cheap to apply are exposed here — the ones that redraw a
table or a figure without recomputing a network. :data:`CONTROLS` is the list.

The server is the standard library's, bound to localhost: a single-user tool,
not a deployment.
"""

from __future__ import annotations

import copy
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import yaml

from pysyrev.core.report.html import HTMLReportEngine, _script_json


# =============================================================================
# The settings exposed
# =============================================================================
#
# section key (its name under topic_report.sections) -> groups of controls.
# A control addresses one config field by its dotted path below the section,
# and says how to edit it:
#   int / float : a number, within [min, max]
#   select      : one of `options`
#   multi       : any of `options`, kept in that order
#   range       : a [low, high] pair of numbers

def _edge_controls(panel: str) -> List[dict]:
    return [
        {"path": f"{panel}.color_by", "kind": "select",
         "options": ["topic", "community"], "hint": "node colour"},
        {"path": f"{panel}.backbone_k", "kind": "int", "min": 1, "max": 20,
         "hint": "strongest links drawn per node"},
        {"path": f"{panel}.edge_scale", "kind": "select",
         "options": ["tertile", "linear"], "hint": "edge width scale"},
        {"path": f"{panel}.edge_width_range", "kind": "range", "min": 0.05,
         "max": 20, "step": 0.1, "hint": "[min, max] width, px — linear scale only"},
    ]


CONTROLS: Dict[str, List[dict]] = {
    "actors": [
        {"title": "Rankings", "controls": [
            {"path": "levels", "kind": "multi",
             "options": ["country", "institution", "author"], "hint": "levels shown"},
            {"path": "top_n", "kind": "int", "min": 1, "max": 500,
             "hint": "actors per ranking"},
            {"path": "sort_by", "kind": "select",
             "options": ["papers", "fractional", "citations"], "hint": "ranking criterion"},
        ]},
        {"title": "Collaboration network", "controls": [
            {"path": "network.size_by", "kind": "select",
             "options": ["citations", "papers", "fractional"], "hint": "node area"},
            {"path": "network.size_range", "kind": "range", "min": 1, "max": 150,
             "step": 1, "hint": "[min, max] node diameter, px"},
            {"path": "network.edge_width_by", "kind": "select",
             "options": ["count", "weight"], "hint": "edge width"},
            {"path": "network.edge_width_range", "kind": "range", "min": 0.05,
             "max": 30, "step": 0.1, "hint": "[min, max] edge width, px"},
            {"path": "network.backbone_k", "kind": "int", "min": 1, "max": 20,
             "hint": "strongest links drawn per node"},
            {"path": "network.n_labels", "kind": "int", "min": 0, "max": 500,
             "hint": "largest nodes named"},
            {"path": "network.node_opacity", "kind": "float", "min": 0.05, "max": 1,
             "step": 0.05, "hint": "1 hides the edges under the nodes"},
        ]},
    ],
    "bib_network": [
        {"title": "Bibliographic coupling", "controls": _edge_controls("coupling")},
        {"title": "Co-citation", "controls": _edge_controls("cocitation")},
        {"title": "Connectivity", "controls": [
            {"path": "connectivity.min_support", "kind": "int", "min": 1, "max": 100000,
             "hint": "observations needed to report an internal ratio"},
        ]},
    ],
}


def _controls(key: str) -> List[dict]:
    return [c for group in CONTROLS.get(key, []) for c in group["controls"]]


# =============================================================================
# Reading and writing config fields by path
# =============================================================================

def _get(cfg: Any, path: str) -> Any:
    for name in path.split("."):
        cfg = getattr(cfg, name)
    return cfg


def _number(value: Any, control: dict, integer: bool) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{control['path']}: expected a number, got {value!r}")
    if integer:
        if int(value) != value:
            raise ValueError(f"{control['path']}: expected a whole number, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if not control["min"] <= value <= control["max"]:
        raise ValueError(f"{control['path']}: {value} is outside "
                         f"[{control['min']}, {control['max']}]")
    return value


def _coerce(value: Any, control: dict) -> Any:
    """*value*, as received from the page, checked against its control."""
    kind, path = control["kind"], control["path"]
    if kind == "int":
        return _number(value, control, integer=True)
    if kind == "float":
        return _number(value, control, integer=False)
    if kind == "select":
        if value not in control["options"]:
            raise ValueError(f"{path}: expected one of {control['options']}, got {value!r}")
        return value
    if kind == "multi":
        if not isinstance(value, list) or any(v not in control["options"] for v in value):
            raise ValueError(f"{path}: expected a list among {control['options']}")
        return [o for o in control["options"] if o in value]
    if kind == "range":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{path}: expected [min, max]")
        low, high = (_number(v, control, integer=False) for v in value)
        if not low < high:
            raise ValueError(f"{path}: min must be below max")
        return [low, high]
    raise ValueError(f"{path}: unknown control kind {kind!r}")


def _apply(section_cfg: Any, key: str, values: Dict[str, Any]) -> Any:
    """A copy of *section_cfg* with *values* (path -> value) set and validated.

    Only the paths :data:`CONTROLS` lists for *key* are accepted. The config
    classes' own checks then run again, innermost block first, so a combination
    the YAML would refuse is refused here too.
    """
    controls = {c["path"]: c for c in _controls(key)}
    unknown = sorted(set(values) - set(controls))
    if unknown:
        raise ValueError(f"not a live setting of {key!r}: {', '.join(unknown)}")
    cfg = copy.deepcopy(section_cfg)
    touched = {""}
    for path, value in values.items():
        parent, _, name = path.rpartition(".")
        setattr(_get(cfg, parent) if parent else cfg, name,
                _coerce(value, controls[path]))
        while parent:
            touched.add(parent)
            parent = parent.rpartition(".")[0]
    for path in sorted(touched, key=lambda p: -len(p.split(".")) if p else 0):
        block = _get(cfg, path) if path else cfg
        if hasattr(block, "__post_init__"):
            block.__post_init__()
    return cfg


# =============================================================================
# Live report
# =============================================================================

class LiveReport:
    """A report whose sections can be rebuilt with other settings.

    Parameters
    ----------
    report_data : dict
        As returned by :func:`~pysyrev.core.report.data.build_report_data`.
    sections_config
        The ``topic_report.sections`` config the report was built with.
    rebuilders : dict
        ``key -> build(section_config)``, as filled by ``build_report_data``.
        A section is live when it has both a rebuilder and controls.
    """

    def __init__(self, report_data: Dict[str, Any], sections_config: Any,
                 rebuilders: Dict[str, Callable], engine: Optional[HTMLReportEngine] = None):
        self.report_data = report_data
        self.engine = engine or HTMLReportEngine()
        self.rebuilders = rebuilders
        present = {s.get("key") for s in report_data.get("sections", [])}
        self.keys = [k for k in CONTROLS if k in rebuilders and k in present]
        self.original = {k: copy.deepcopy(getattr(sections_config, k)) for k in self.keys}
        self.config = copy.deepcopy(self.original)
        self._lock = threading.Lock()
        self._page: Optional[str] = None

    # ---- state ------------------------------------------------------------

    def controls(self) -> Dict[str, List[dict]]:
        """The panel's content: each live section's groups, with for every
        control its current ``value`` and the ``default`` it started from."""
        out = {}
        for key in self.keys:
            groups = copy.deepcopy(CONTROLS[key])
            for control in (c for g in groups for c in g["controls"]):
                control["value"] = _get(self.config[key], control["path"])
                control["default"] = _get(self.original[key], control["path"])
            out[key] = groups
        return out

    def changes(self) -> Dict[str, Any]:
        """The settings that differ from the config, nested as in the YAML."""
        sections: Dict[str, Any] = {}
        for key in self.keys:
            for control in _controls(key):
                value = _get(self.config[key], control["path"])
                if value == _get(self.original[key], control["path"]):
                    continue
                node = sections.setdefault(key, {})
                *parents, name = control["path"].split(".")
                for parent in parents:
                    node = node.setdefault(parent, {})
                node[name] = value
        return {"topic_report": {"sections": sections}} if sections else {}

    def yaml(self) -> str:
        changes = self.changes()
        if not changes:
            return "# no change yet"
        return yaml.safe_dump(changes, sort_keys=False, default_flow_style=None)

    # ---- rendering --------------------------------------------------------

    def page(self) -> str:
        """The whole page, in its current state."""
        with self._lock:
            if self._page is None:
                spec = json.dumps({"controls": self.controls(), "yaml": self.yaml()})
                self._page = self.engine.render(
                    self.report_data,
                    extra_css=_LIVE_CSS,
                    extra_html=_PANEL_HTML % _script_json(spec),
                    extra_js=_LIVE_JS,
                    force_plotlyjs=True)
            return self._page

    def update(self, key: str, values: Dict[str, Any]) -> Dict[str, str]:
        """Rebuild section *key* with *values* (control path -> value).

        Returns the fragments the page swaps in — ``html`` (the section's
        content), ``subnav`` (its sidebar links) — and the ``yaml`` of all the
        changes so far. Raises ``ValueError`` on a setting the config refuses,
        leaving the report as it was.
        """
        if key not in self.keys:
            raise ValueError(f"section {key!r} has no live setting")
        if not isinstance(values, dict):
            raise ValueError("expected an object of setting -> value")
        with self._lock:
            cfg = _apply(self.config[key], key, values)
            section = self.rebuilders[key](cfg)
            if section is None:
                raise ValueError(f"section {key!r} came out empty with these settings")
            section["key"] = key
            sections = self.report_data["sections"]
            index = next(i for i, s in enumerate(sections) if s.get("key") == key)
            sections[index] = section
            self.config[key] = cfg
            self._page = None
            html, subnav = self.engine.render_section(section, f"section-{index + 1}")
            return {"html": html, "subnav": subnav, "yaml": self.yaml()}

    # ---- serving ----------------------------------------------------------

    def make_server(self, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
        """An HTTP server for this report; falls back to a free port when
        *port* is taken."""
        try:
            server = ThreadingHTTPServer((host, port), _Handler)
        except OSError:
            server = ThreadingHTTPServer((host, 0), _Handler)
        server.live = self
        return server

    def serve(self, host: str = "127.0.0.1", port: int = 8765,
              open_browser: bool = True) -> None:
        """Serve the report until interrupted (Ctrl+C)."""
        server = self.make_server(host, port)
        url = f"http://{host}:{server.server_address[1]}/"
        print(f"Live report on {url} — Ctrl+C to stop.")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLive report stopped.")
        finally:
            server.server_close()


class _Handler(BaseHTTPRequestHandler):
    """``GET /`` is the page; ``POST /api/section/<key>`` rebuilds a section."""

    def log_message(self, format, *args):   # noqa: A002 - signature of the base class
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):   # noqa: N802
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, self.server.live.page().encode("utf-8"),
                       "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):   # noqa: N802
        prefix = "/api/section/"
        if not self.path.startswith(prefix):
            return self._send(404, b"Not found", "text/plain")
        # A JSON body cannot be posted by another site's page without a
        # preflight this server never answers.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._json(415, {"error": "expected application/json"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            values = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, self.server.live.update(self.path[len(prefix):], values))
        except ValueError as e:      # a refused setting, or a body that is not JSON
            self._json(400, {"error": str(e)})
        except Exception as e:       # the rebuild itself failed: say so, stay up
            self._json(500, {"error": f"rebuild failed: {e!r}"})


# =============================================================================
# Page assets
# =============================================================================

_PANEL_HTML = """<aside class="settings" id="settings">
<h2>Settings</h2>
<div class="settings-status" id="settings-status"></div>
<div id="settings-body"></div>
<div class="settings-none" id="settings-none">No live setting for this section.</div>
<div class="settings-yaml">
<h3>Changed settings <button type="button" id="settings-copy">Copy</button></h3>
<pre id="settings-yaml"></pre>
<p>Paste into the config to keep them.</p>
</div>
<script type="application/json" id="live-spec">%s</script>
</aside>
"""

_LIVE_CSS = """
.settings {
  flex: 0 0 310px; width: 310px; position: sticky; top: 0; height: 100vh;
  overflow-y: auto; padding: 22px 18px 30px; background: #f7f9fc;
  border-left: 1px solid var(--line); font-size: 13px;
}
.settings h2 { font-size: 17px; color: var(--primary); margin: 0 0 4px; }
.settings h3 {
  font-size: 13px; color: var(--secondary); margin: 18px 0 8px; padding-bottom: 4px;
  border-bottom: 1px solid var(--line); text-transform: uppercase; letter-spacing: .04em;
}
.settings-status { min-height: 18px; font-size: 12px; color: var(--muted); }
.settings-status.error { color: #c0392b; }
.settings-none { color: var(--muted); margin-top: 14px; }
.settings .control { margin: 0 0 11px; }
.settings .control > label, .settings .control > .name {
  display: block; font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 12px; font-weight: 600; margin-bottom: 2px;
}
.settings .control.changed > label, .settings .control.changed > .name { color: var(--link); }
.settings .hint { color: var(--muted); font-size: 11.5px; margin-top: 2px; }
.settings input[type=number], .settings select {
  font: inherit; padding: 4px 6px; border: 1px solid var(--grid);
  border-radius: 4px; background: #fff; width: 100%;
}
.settings .pair { display: flex; gap: 6px; }
.settings .checks label { display: inline-block; margin-right: 10px; font-weight: 400; }
.settings button {
  font: inherit; font-size: 12px; padding: 3px 9px; cursor: pointer;
  border: 1px solid var(--grid); border-radius: 4px; background: #fff;
}
.settings .reset { margin-top: 4px; }
.settings-yaml h3 button { float: right; text-transform: none; letter-spacing: 0; }
.settings-yaml pre { font-size: 12px; margin: 0 0 6px; white-space: pre-wrap; }
.settings-yaml p { color: var(--muted); font-size: 11.5px; margin: 0; }
.section.updating { opacity: .45; transition: opacity .15s; }
@media (max-width: 1200px) {
  .layout { flex-wrap: wrap; }
  .settings { position: static; height: auto; flex-basis: 100%; width: auto; order: -1; }
}
"""

_LIVE_JS = """
(function () {
  var spec = JSON.parse(document.getElementById('live-spec').textContent);
  var body = document.getElementById('settings-body');
  var none = document.getElementById('settings-none');
  var status = document.getElementById('settings-status');
  var yamlBox = document.getElementById('settings-yaml');
  var readers = {};     // section key -> [function () -> [path, value]]
  var groups = {};      // section key -> its controls' element
  var timers = {}, requests = {};

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined) { e.textContent = text; }
    return e;
  }
  function numberInput(c, value) {
    var i = el('input');
    i.type = 'number'; i.value = value;
    if (c.min !== undefined) { i.min = c.min; }
    if (c.max !== undefined) { i.max = c.max; }
    i.step = c.step !== undefined ? c.step : (c.kind === 'int' ? 1 : 'any');
    return i;
  }

  // One control: its element, how to read it, how to set it.
  function build(c, onChange) {
    var box = el('div', 'control'), read, write;
    box.appendChild(el(c.kind === 'multi' || c.kind === 'range' ? 'div' : 'label',
                       c.kind === 'multi' || c.kind === 'range' ? 'name' : '', c.path));
    if (c.kind === 'select') {
      var s = el('select');
      c.options.forEach(function (o) { var op = el('option', '', o); op.value = o; s.appendChild(op); });
      box.appendChild(s);
      read = function () { return s.value; };
      write = function (v) { s.value = v; };
    } else if (c.kind === 'multi') {
      var checks = el('div', 'checks'), inputs = [];
      c.options.forEach(function (o) {
        var l = el('label'), i = el('input');
        i.type = 'checkbox'; i.value = o;
        l.appendChild(i); l.appendChild(document.createTextNode(' ' + o));
        checks.appendChild(l); inputs.push(i);
      });
      box.appendChild(checks);
      read = function () {
        return inputs.filter(function (i) { return i.checked; }).map(function (i) { return i.value; });
      };
      write = function (v) { inputs.forEach(function (i) { i.checked = v.indexOf(i.value) !== -1; }); };
    } else if (c.kind === 'range') {
      var pair = el('div', 'pair'), lo = numberInput(c, 0), hi = numberInput(c, 0);
      pair.appendChild(lo); pair.appendChild(hi); box.appendChild(pair);
      read = function () { return [parseFloat(lo.value), parseFloat(hi.value)]; };
      write = function (v) { lo.value = v[0]; hi.value = v[1]; };
    } else {
      var n = numberInput(c, 0);
      box.appendChild(n);
      read = function () { return parseFloat(n.value); };
      write = function (v) { n.value = v; };
    }
    if (c.hint) { box.appendChild(el('div', 'hint', c.hint)); }
    function mark() {
      box.classList.toggle('changed', JSON.stringify(read()) !== JSON.stringify(c['default']));
    }
    write(c.value); mark();
    box.addEventListener('input', function () { mark(); onChange(); });
    return {box: box, read: function () { return [c.path, read()]; },
            reset: function () { write(c['default']); mark(); }};
  }

  function send(key) {
    var values = {}, complete = true;
    readers[key].forEach(function (r) {
      var pv = r.read(), flat = [].concat(pv[1]);
      if (flat.some(function (x) { return typeof x === 'number' && isNaN(x); })) { complete = false; }
      values[pv[0]] = pv[1];
    });
    if (!complete) { return; }           // a field being typed: wait for a number
    var section = document.querySelector('.section[data-key="' + key + '"]');
    var id = requests[key] = (requests[key] || 0) + 1;
    status.className = 'settings-status'; status.textContent = 'Updating…';
    section.classList.add('updating');
    fetch('/api/section/' + key, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(values)
    }).then(function (r) {
      return r.json().then(function (j) { return {ok: r.ok, data: j}; });
    }).then(function (res) {
      if (id !== requests[key]) { return; }      // a newer change is on its way
      section.classList.remove('updating');
      if (!res.ok) {
        status.className = 'settings-status error';
        status.textContent = res.data.error || 'Update failed.';
        return;
      }
      if (window.Plotly) {
        section.querySelectorAll('.figure-plot').forEach(function (p) { window.Plotly.purge(p); });
      }
      section.innerHTML = res.data.html;
      document.querySelector('.subnav[data-for="' + section.id + '"]').innerHTML = res.data.subnav;
      window.report.bindTables(section);
      if (section.classList.contains('active')) { window.report.renderFigures(section); }
      yamlBox.textContent = res.data.yaml;
      status.textContent = 'Up to date.';
    }).catch(function () {
      if (id !== requests[key]) { return; }
      section.classList.remove('updating');
      status.className = 'settings-status error';
      status.textContent = 'The report server is not answering — is it still running?';
    });
  }

  Object.keys(spec.controls).forEach(function (key) {
    var root = el('div');
    readers[key] = [];
    function changed() {
      clearTimeout(timers[key]);
      timers[key] = setTimeout(function () { send(key); }, 350);
    }
    spec.controls[key].forEach(function (group) {
      root.appendChild(el('h3', '', group.title));
      group.controls.forEach(function (c) {
        var control = build(c, changed);
        readers[key].push(control);
        root.appendChild(control.box);
      });
    });
    var reset = el('button', 'reset', 'Reset this section');
    reset.type = 'button';
    reset.addEventListener('click', function () {
      readers[key].forEach(function (r) { r.reset(); });
      changed();
    });
    root.appendChild(reset);
    root.style.display = 'none';
    groups[key] = root;
    body.appendChild(root);
  });

  function showFor(section) {
    var key = section && section.getAttribute('data-key'), any = false;
    Object.keys(groups).forEach(function (k) {
      groups[k].style.display = k === key ? '' : 'none';
      any = any || k === key;
    });
    none.style.display = any ? 'none' : '';
  }
  document.addEventListener('report:section', function (e) { showFor(e.detail); });
  showFor(document.querySelector('.section.active'));

  yamlBox.textContent = spec.yaml;
  document.getElementById('settings-copy').addEventListener('click', function () {
    if (navigator.clipboard) { navigator.clipboard.writeText(yamlBox.textContent); }
  });
})();
"""
