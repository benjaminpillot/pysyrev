"""
Tests for pysyrev.core.report.live — the served report and its settings.

Fast: the section rebuilders are stand-ins, except for the actors section,
which is rebuilt for real from the test dataset.
"""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pysyrev.core.config import ActorsSectionConfig, BibNetworkSectionConfig
from pysyrev.core.report import live as live_mod
from pysyrev.core.report.live import CONTROLS, LiveReport, _apply


def _section(key, text):
    return {"title": f"1. {key}", "key": key, "blocks": [
        {"type": "subsection", "title": text, "blocks": [
            {"type": "paragraph", "text": text}]}]}


@pytest.fixture
def live():
    cfg = SimpleNamespace(actors=ActorsSectionConfig(),
                          bib_network=BibNetworkSectionConfig())
    data = {"meta": {"title": "T"},
            "sections": [{"title": "0. Static", "key": "topics", "blocks": []},
                         _section("actors", "top 15")]}
    calls = []

    def rebuild(section_cfg):
        calls.append(section_cfg)
        return _section("actors", f"top {section_cfg.top_n}")

    report = LiveReport(data, cfg, {"actors": rebuild})
    report.calls = calls
    return report


class TestControls:
    @pytest.mark.parametrize("key, cls", [("actors", ActorsSectionConfig),
                                          ("bib_network", BibNetworkSectionConfig)])
    def test_every_control_addresses_a_real_field_with_a_valid_default(self, key, cls):
        cfg = cls()
        values = {c["path"]: live_mod._get(cfg, c["path"])
                  for g in CONTROLS[key] for c in g["controls"]}
        # The defaults pass their own controls, and come back unchanged.
        applied = _apply(cfg, key, values)
        for path, value in values.items():
            assert live_mod._get(applied, path) == value


class TestApply:
    def test_sets_nested_fields_on_a_copy(self):
        cfg = ActorsSectionConfig()
        out = _apply(cfg, "actors", {"top_n": 3, "network.size_by": "papers"})
        assert (out.top_n, out.network.size_by) == (3, "papers")
        assert (cfg.top_n, cfg.network.size_by) == (15, "citations")

    @pytest.mark.parametrize("values, message", [
        ({"network.min_papers": 1}, "not a live setting"),
        ({"top_n": 0}, "outside"),
        ({"top_n": 2.5}, "whole number"),
        ({"top_n": "3"}, "expected a number"),
        ({"sort_by": "nope"}, "expected one of"),
        ({"levels": ["planet"]}, "expected a list"),
        ({"network.size_range": [9, 3]}, "min must be below max"),
        ({"network.size_range": [9]}, r"expected \[min, max\]"),
    ])
    def test_refuses(self, values, message):
        with pytest.raises(ValueError, match=message):
            _apply(ActorsSectionConfig(), "actors", values)

    def test_levels_keep_the_canonical_order(self):
        out = _apply(ActorsSectionConfig(), "actors", {"levels": ["author", "country"]})
        assert out.levels == ["country", "author"]


class TestLiveReport:
    def test_only_sections_with_controls_a_rebuilder_and_a_place_are_live(self, live):
        assert live.keys == ["actors"]           # bib_network: no rebuilder, not in the page

    def test_page_carries_the_panel_and_its_state(self, live):
        page = live.page()
        assert 'id="settings"' in page and 'data-key="actors"' in page
        spec = json.loads(page.split('id="live-spec">')[1].split("</script>")[0])
        top_n = spec["controls"]["actors"][0]["controls"][1]
        assert (top_n["path"], top_n["value"], top_n["default"]) == ("top_n", 15, 15)

    def test_update_rebuilds_the_section_and_reports_the_change(self, live):
        out = live.update("actors", {"top_n": 4, "network.size_by": "papers"})
        assert "top 4" in out["html"] and "top 4" in out["subnav"]
        assert 'id="section-2-1"' in out["html"]      # the section kept its place
        assert out["yaml"] == ("topic_report:\n  sections:\n    actors:\n"
                               "      top_n: 4\n      network: {size_by: papers}\n")
        assert live.calls[-1].top_n == 4
        # The page reflects it too — a reload does not lose the settings.
        assert "top 4" in live.page() and "top 15" not in live.page()

    def test_changes_accumulate_and_undo(self, live):
        live.update("actors", {"top_n": 4})
        live.update("actors", {"sort_by": "citations"})
        assert live.changes() == {"topic_report": {"sections": {
            "actors": {"top_n": 4, "sort_by": "citations"}}}}
        live.update("actors", {"top_n": 15, "sort_by": "papers"})
        assert live.changes() == {} and live.yaml() == "# no change yet"

    def test_a_refused_setting_leaves_the_report_as_it_was(self, live):
        with pytest.raises(ValueError):
            live.update("actors", {"top_n": 0})
        assert live.calls == [] and "top 15" in live.page()

    def test_unknown_section(self, live):
        with pytest.raises(ValueError, match="no live setting"):
            live.update("topics", {})


class TestServer:
    @pytest.fixture
    def url(self, live):
        server = live.make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()
        server.server_close()

    def _post(self, url, body, content_type="application/json"):
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_serves_the_page(self, url):
        with urllib.request.urlopen(url + "/") as r:
            assert r.status == 200 and b'id="settings"' in r.read()

    def test_update_round_trip(self, url):
        status, out = self._post(url + "/api/section/actors", {"top_n": 7})
        assert status == 200 and "top 7" in out["html"]

    def test_a_refused_setting_is_a_400_with_its_reason(self, url):
        status, out = self._post(url + "/api/section/actors", {"sort_by": "nope"})
        assert status == 400 and "expected one of" in out["error"]

    def test_only_json_bodies_are_accepted(self, url):
        status, _ = self._post(url + "/api/section/actors", {"top_n": 7},
                               content_type="text/plain")
        assert status == 415

    def test_a_failing_rebuild_does_not_take_the_server_down(self, url, live):
        def boom(cfg):
            raise RuntimeError("no data")

        live.rebuilders["actors"] = boom
        status, out = self._post(url + "/api/section/actors", {"top_n": 7})
        assert status == 500 and "rebuild failed" in out["error"]
        with urllib.request.urlopen(url + "/") as r:
            assert r.status == 200


class TestActorsRebuild:
    """The real actors section, rebuilt with other settings."""

    def test_network_is_reused_when_its_settings_do_not_change(self, monkeypatch):
        import pysyrev.core.networks as networks
        from pysyrev.core.report.sections.actors import _build_actors_section

        df = pd.read_csv(Path(__file__).parent / "data" / "bib_dataset.csv",
                         low_memory=False)
        built = []
        real = networks.build_collaboration

        def counting(*args, **kwargs):
            built.append(args[1])
            return real(*args, **kwargs)

        monkeypatch.setattr(networks, "build_collaboration", counting)
        cache = {}
        cfg = ActorsSectionConfig(levels=["country"])
        first = _build_actors_section(df, cfg, None, 1, network_cache=cache)
        again = _build_actors_section(
            df, _apply(cfg, "actors", {"top_n": 3, "network.size_by": "papers"}),
            None, 1, network_cache=cache)
        assert built == ["country"]
        assert len(first["blocks"][-1]["blocks"][0]["rows"]) > 3
        assert len(again["blocks"][-1]["blocks"][0]["rows"]) == 3

        cfg.network.min_papers = 1               # a setting the network depends on
        _build_actors_section(df, cfg, None, 1, network_cache=cache)
        assert built == ["country", "country"]
