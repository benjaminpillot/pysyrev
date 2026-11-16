"""A stage is only as fragile as its own config section.

Running one stage must not be held hostage by a section it never reads: an
incomplete `topic_model:` cannot stop `--stage bib`. The section's error is
kept, not thrown away, and surfaces the moment a stage that needs it runs —
naming the mandatory key by its dotted YAML path.
"""

import textwrap

import pytest

from pysyrev.core.config import (SECTION_CLASSES, Config, ConfigError,
                                 mandatory_keys)
from pysyrev.pipeline import Pipeline


def _write(tmp_path, *parts: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text("".join(textwrap.dedent(p) for p in parts), encoding="utf-8")
    return str(path)


def _review_section(tmp_path) -> str:
    # Every mandatory key of `review:`, and none of the optional ones — which
    # doubles as a check that the model knobs a reviewer may leave blank (host,
    # max_tokens, temperature, reasoning_effort…) are not among them.
    return f"""
        review:
          export:
            export_dir: {tmp_path}/review
          text_inputs: [title, abstract]
          inclusion_criteria: about X
          exclusion_criteria: not about Y
          reviewers:
            - name: R1
              model_id: m
              provider: anthropic
              backstory: an expert
          workflow:
            - round: A
              reviewers: [R1]
        """


@pytest.fixture
def stub_stages(monkeypatch):
    """Neutralise the stage runtimes: this is about config, not about work."""
    import pysyrev.bibdata as bibdata_mod
    import pysyrev.review as review_mod

    class _Bib:
        dataset = []

    class _Review:
        export = type("E", (), {"included_docs": "/runs/new/reviewed_included.csv"})()
        def run(self, dataset): return self
        def save(self): return self

    monkeypatch.setattr(bibdata_mod.BibDataset, "from_config",
                        classmethod(lambda cls, cfg: _Bib()))
    monkeypatch.setattr(review_mod.LLMReview, "from_config",
                        classmethod(lambda cls, cfg: _Review()))


# ── An unread section cannot stop a stage ───────────────────────────────────

class TestABrokenSectionOnlyStopsItsOwnStages:

    @pytest.fixture
    def config_file(self, tmp_path):
        return _write(tmp_path, f"""
            bib:
              open_alex: {tmp_path}/corpus.csv
              export:
                export_dir: {tmp_path}/bib
            topic_model:
              desired_topics: [5, 15]
            """)

    def test_loading_does_not_raise(self, config_file, capsys):
        config = Config.load(config_file)
        assert config.bib is not None
        assert config.topic_model is None            # unusable, so not handed out
        assert "topic_model" in config.errors
        assert "unusable" in capsys.readouterr().out  # never silently

    def test_the_bib_stage_still_runs(self, config_file, stub_stages):
        Pipeline.from_config(config_file).run(stages=["bib"])   # must not raise

    def test_the_stage_that_needs_it_raises_naming_the_key(self, config_file):
        with pytest.raises(ConfigError, match=r"topic_model\.export"):
            Pipeline.from_config(config_file).run(stages=["topic-model"])

    def test_a_full_run_still_fails(self, config_file, stub_stages):
        # Dropping the broken stage here would quietly turn a full run into a
        # shorter one — the failure the user asked for is the loud one.
        with pytest.raises(ConfigError, match=r"topic_model\.export"):
            Pipeline.from_config(config_file).run()


class TestASectionAStageOnlyUsesNeverStopsIt:

    def test_the_review_stage_runs_and_warns(self, tmp_path, stub_stages, capsys):
        config_file = _write(tmp_path, _review_section(tmp_path), """
            topic_model:
              desired_topics: [5, 15]
            """)
        Pipeline.from_config(config_file).run(stages=["review"])
        assert "runs without `topic_model:`" in capsys.readouterr().out


# ── Telling the user what is mandatory ──────────────────────────────────────

class TestMandatoryKeysAreNamed:

    def test_an_absent_section_names_what_it_would_need(self, tmp_path):
        config_file = _write(tmp_path, f"""
            bib:
              open_alex: {tmp_path}/corpus.csv
            """)
        with pytest.raises(ConfigError,
                           match=r"needs a `topic_model:` section.*"
                                 r"topic_model\.export\.export_dir"):
            Config.load(config_file).require_stages(["topic-model"])

    def test_a_section_with_nothing_mandatory_says_so(self, tmp_path):
        config_file = _write(tmp_path, f"""
            bib:
              open_alex: {tmp_path}/corpus.csv
            """)
        with pytest.raises(ConfigError, match="Mandatory keys: none"):
            Config.load(config_file).require_stages(["topic-report"])

    def test_an_unknown_key_is_named_by_its_path(self, tmp_path, capsys):
        config_file = _write(tmp_path, """
            topic_model:
              export: {export_dir: /tmp/x}
              desired_topic: [5, 15]
            """)
        config = Config.load(config_file)
        assert "topic_model.desired_topic" in str(config.errors["topic_model"])

    def test_an_unknown_section_is_not_silently_dropped(self, tmp_path, capsys):
        # A typo in a section name would otherwise read as "stage not configured".
        config_file = _write(tmp_path, f"""
            bib:
              open_alex: {tmp_path}/corpus.csv
            bib_network:
              coupling_network: {{}}
            """)
        Config.load(config_file)
        assert "unknown section `bib_network:`" in capsys.readouterr().out

    def test_nested_mandatory_keys_are_walked(self):
        keys = mandatory_keys(SECTION_CLASSES["review"], "review")
        assert "review.export.export_dir" in keys
        assert "review.reviewers[].model_id" in keys
        # A knob the runtime happily omits must not be advertised as mandatory.
        assert not any(k.endswith((".host", ".max_tokens", ".temperature"))
                       for k in keys)

    def test_a_section_with_no_mandatory_key_reports_none(self):
        assert mandatory_keys(SECTION_CLASSES["bib"], "bib") == []
