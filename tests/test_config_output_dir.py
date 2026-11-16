"""One root `output_dir` places every section's results under one project folder.

Each section exports to ``<output_dir>/<section>/`` unless its own
``export.export_dir`` overrides it: a relative override is taken under the root,
an absolute one is used as is. The stages then find each other's latest run
under that root — even when the upstream section is absent from the config.
"""

import os
import textwrap

from pysyrev.core.config import Config


class _Yaml(str):
    """A YAML snippet; ``+`` dedents each side on its own before joining."""
    def __add__(self, other):
        return _Yaml(textwrap.dedent(self) + textwrap.dedent(other))


def _write(tmp_path, body: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


_REVIEW = """
    review:
      text_inputs: [title, abstract]
      inclusion_criteria: about X
      exclusion_criteria: not about Y
      reviewers:
        - {name: R1, model_id: m, provider: anthropic, backstory: an expert}
      workflow:
        - {round: A, reviewers: [R1]}
"""


class TestEachSectionGetsASubfolder:

    def test_blank_export_dirs_default_to_the_section_name(self, tmp_path):
        root = tmp_path / "project"
        config = Config.load(_write(tmp_path, _Yaml(f"""
            output_dir: {root}
            bib:
              open_alex: {tmp_path}/corpus.csv
              export: {{}}
            topic_model: {{desired_topics: [5, 15]}}
            topic_report: {{meta: {{title: T}}}}
            """) + _REVIEW))
        assert config.errors == {}
        assert config.output_dir == str(root)
        assert config.bib.export.export_dir == str(root / "bib")
        assert config.review.export.export_dir == str(root / "review")
        assert config.topic_model.export.export_dir == str(root / "topic_model")
        assert config.topic_report.export.export_dir == str(root / "topic_report")

    def test_a_relative_override_stays_under_the_root(self, tmp_path):
        config = Config.load(_write(tmp_path, _Yaml(f"""
            output_dir: {tmp_path}/project
            topic_model:
              export: {{export_dir: tm_specter}}
            """)))
        assert config.topic_model.export.export_dir == \
            os.path.join(str(tmp_path), "project", "tm_specter")

    def test_an_absolute_override_is_kept(self, tmp_path):
        config = Config.load(_write(tmp_path, _Yaml(f"""
            output_dir: {tmp_path}/project
            topic_model:
              export: {{export_dir: {tmp_path}/elsewhere}}
            """)))
        assert config.topic_model.export.export_dir == f"{tmp_path}/elsewhere"

    def test_without_a_root_the_section_paths_are_untouched(self, tmp_path):
        config = Config.load(_write(tmp_path, """
            topic_model:
              export: {export_dir: relative/dir}
            """))
        assert config.output_dir is None
        assert config.topic_model.export.export_dir == "relative/dir"

    def test_without_a_root_a_missing_export_dir_suggests_one(self, tmp_path):
        config = Config.load(_write(tmp_path, """
            topic_model: {desired_topics: [5, 15]}
            """))
        assert "output_dir" in str(config.errors["topic_model"])


class TestStagesFindEachOtherUnderTheRoot:

    def test_review_reads_the_latest_bib_run_without_a_bib_section(self, tmp_path):
        run = tmp_path / "project" / "bib" / "2026-01-01T000000"
        run.mkdir(parents=True)
        (run / "bib_dataset.csv").write_text("title\n")
        config = Config.load(_write(tmp_path, _Yaml(f"""
            output_dir: {tmp_path}/project
            """) + _REVIEW))
        assert config.review.doc_dataset == str(run / "bib_dataset.csv")

    def test_report_reads_the_latest_topic_model_run(self, tmp_path):
        run = tmp_path / "project" / "topic_model" / "run_1"
        run.mkdir(parents=True)
        config = Config.load(_write(tmp_path, _Yaml(f"""
            output_dir: {tmp_path}/project
            topic_report: {{meta: {{title: T}}}}
            """)))
        assert config.topic_report.run_dir == str(run)


class TestLegacyExportTo:

    def test_export_to_still_works_and_warns(self, tmp_path, capsys):
        config = Config.load(_write(tmp_path, _Yaml(f"""
            topic_report:
              export_to: {tmp_path}/report
            """)))
        assert config.topic_report.export.export_dir == f"{tmp_path}/report"
        assert "deprecated" in capsys.readouterr().out


class TestABlankKeyFallsBackToItsDefault:
    """`key:` with no value is YAML for None — it must not erase the default."""

    def test_blank_reasoning_is_brief(self, tmp_path):
        config = Config.load(_write(tmp_path, """
            review:
              export: {export_dir: /tmp/x}
              text_inputs: [title]
              inclusion_criteria: a
              exclusion_criteria: b
              batch_size:
              reviewers:
                - {name: R1, model_id: m, provider: albert, backstory: s, reasoning: }
              workflow:
                - {round: A, reviewers: [R1]}
            """))
        assert config.review.reviewers[0].reasoning == "brief"
        assert config.review.batch_size == 100

    def test_blank_extract_year_is_the_default_floor(self, tmp_path):
        config = Config.load(_write(tmp_path, f"""
            bib:
              open_alex: {tmp_path}/corpus.csv
              extract:
                year:
            """))
        assert config.bib.extract.year == 1900
