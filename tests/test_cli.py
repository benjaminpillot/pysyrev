"""
CLI integration tests for `python -m pysyrev`.

Fast tests (no mark): argument parsing, error cases, --help.
Integration tests (marked `integration`): end-to-end stage execution via
subprocess, reusing the session-scoped topic_model_outputs fixture.

Run with:
    pytest tests/test_cli.py                    # fast only
    pytest tests/test_cli.py -m integration     # all
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# =============================================================================
# Helpers
# =============================================================================

def _run(*args, cwd=None):
    """Run `python -m pysyrev <args>` and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "pysyrev", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _write_config(path: Path, content: str) -> Path:
    cfg = path / "config.yaml"
    cfg.write_text(textwrap.dedent(content))
    return cfg


# =============================================================================
# Fast tests — argument parsing and error cases (no ML, no fixtures)
# =============================================================================

class TestCLIArguments:

    def test_help_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0

    def test_help_lists_all_stages(self):
        result = _run("--help")
        for stage in ("bib", "review", "topic-model", "topic-report"):
            assert stage in result.stdout

    def test_missing_config_file_exits_nonzero(self):
        result = _run("nonexistent_config.yaml")
        assert result.returncode != 0

    def test_invalid_stage_exits_nonzero(self, tmp_path):
        cfg = _write_config(tmp_path, "topic_report: {}\n")
        result = _run(str(cfg), "--stage", "invalid-stage")
        assert result.returncode != 0

    def test_stage_and_from_are_mutually_exclusive(self, tmp_path):
        cfg = _write_config(tmp_path, "# no sections\n")
        result = _run(str(cfg), "--stage", "bib", "--from", "review")
        assert result.returncode != 0

    def test_invalid_from_stage_exits_nonzero(self, tmp_path):
        cfg = _write_config(tmp_path, "# no sections\n")
        result = _run(str(cfg), "--from", "invalid-stage")
        assert result.returncode != 0


# =============================================================================
# Integration tests — real execution via subprocess
# =============================================================================

@pytest.mark.integration
class TestCLITopicReport:

    def test_topic_report_stage_exits_zero(self, tmp_path, topic_model_outputs):
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {tmp_path / "report"}
        """)
        result = _run(str(cfg), "--stage", "topic-report")
        assert result.returncode == 0, result.stderr

    def test_topic_report_stage_prints_done(self, tmp_path, topic_model_outputs):
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {tmp_path / "report"}
        """)
        result = _run(str(cfg), "--stage", "topic-report")
        assert "[topic-report] Done" in result.stdout

    def test_topic_report_stage_produces_pdf(self, tmp_path, topic_model_outputs):
        export_dir = tmp_path / "report"
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {export_dir}
        """)
        _run(str(cfg), "--stage", "topic-report")
        pdfs = list(export_dir.glob("**/*.pdf"))
        assert len(pdfs) == 1

    def test_all_runs_only_configured_stages(self, tmp_path, topic_model_outputs):
        """'all' with only topic_report in config must not fail on absent stages."""
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {tmp_path / "report"}
        """)
        result = _run(str(cfg))
        assert result.returncode == 0, result.stderr

    def test_missing_topic_report_section_skipped(self, tmp_path, topic_model_outputs):
        """A config with no topic_report section must succeed and skip the stage."""
        cfg = _write_config(tmp_path, "# no sections\n")
        result = _run(str(cfg))
        assert result.returncode == 0
        assert "[topic-report]" not in result.stdout

    def test_multi_stage_runs_both(self, tmp_path, topic_model_outputs):
        """--stage with two values must run both stages."""
        export_dir = tmp_path / "report"
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {export_dir}
        """)
        result = _run(str(cfg), "--stage", "topic-report", "topic-report")
        assert result.returncode == 0, result.stderr

    def test_from_stage_runs_configured_stages(self, tmp_path, topic_model_outputs):
        """--from topic-report must run topic-report when it is the only configured stage."""
        export_dir = tmp_path / "report"
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {export_dir}
        """)
        result = _run(str(cfg), "--from", "topic-report")
        assert result.returncode == 0, result.stderr
        assert "[topic-report] Done" in result.stdout

    def test_from_stage_skips_earlier_unconfigured_stages(self, tmp_path, topic_model_outputs):
        """--from topic-model must not attempt bib or review when absent from config."""
        export_dir = tmp_path / "report"
        cfg = _write_config(tmp_path, f"""\
            topic_report:
              run_dir: {topic_model_outputs["run_dir"]}
              export_to: {export_dir}
        """)
        result = _run(str(cfg), "--from", "topic-model")
        assert result.returncode == 0, result.stderr
        assert "[topic-report] Done" in result.stdout
