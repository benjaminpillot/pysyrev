"""
pysyrev CLI — systematic literature review pipeline.

Usage
-----
  # Run all configured pipeline stages
  pysyrev config.yaml

  # Run one or more specific stages
  pysyrev config.yaml --stage topic-model
  pysyrev config.yaml --stage topic-model topic-report

  # Run from a given stage to the end (all configured stages from that point)
  pysyrev config.yaml --from topic-model

  # Download full-text papers for a list of candidates
  pysyrev download liste.csv output_folder [--config download_config.yaml]

  # Explore the report live: the interactive page plus a settings panel
  pysyrev serve config.yaml [--port 8765] [--no-browser]

  # Price the review stage before paying for it (no model call)
  pysyrev estimate config.yaml [--calibrate reviewed.csv] [--sweep]

  # List the mandatory sections/keys, and check a config against them
  pysyrev check [config.yaml] [--stage bib]

The ``download`` subcommand tries to retrieve each paper in cascade order:
Unpaywall → OpenAlex → Elsevier TDM.  Passing ``--config`` injects API keys
and fine-grained options; without it, only the OpenAlex step runs (no key
required).  The positional arguments always override ``doc_dataset`` and
``output_dir`` from the config file.
"""

import argparse
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from pysyrev import ALL_STAGES, Pipeline, __version__


# =============================================================================
# Pipeline sub-command
# =============================================================================

def _print_stage_result(stage, pipeline):
    if stage == 'bib' and pipeline.bib is not None:
        print(f"[bib] Done — {len(pipeline.bib.dataset)} documents.")
    elif stage == 'review' and pipeline.review is not None:
        print(f"[review] Done — {len(pipeline.review.included_docs)} documents included.")
    elif stage == 'topic-model':
        print("[topic-model] Done.")
    elif stage == 'topic-report' and pipeline.report is not None:
        print(f"[topic-report] Done — report written to {pipeline.report.export_to}")


def _run_pipeline(argv):
    parser = argparse.ArgumentParser(
        prog='pysyrev',
        description='Systematic literature review pipeline.',
        epilog=(
            'sub-commands (run `pysyrev <sub-command> --help` for their options):\n'
            '  serve     config.yaml   explore the report live: the interactive page\n'
            '                          plus a panel of settings, served locally\n'
            '  estimate  config.yaml   price the review stage before running it\n'
            '                          (no model call)\n'
            '  check     [config.yaml] list the mandatory keys, or check a config\n'
            '                          against them\n'
            '  download  list.csv dir  download the full texts of a list of papers\n'
            '                          (Unpaywall -> OpenAlex -> Elsevier TDM)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('config', help='Path to the YAML pipeline config file.')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--stage',
        nargs='+',
        choices=ALL_STAGES,
        metavar='STAGE',
        help=(
            'One or more stages to run: '
            'bib | review | topic-model | topic-report. '
            'Stages are always executed in canonical order.'
        ),
    )
    mode.add_argument(
        '--from',
        dest='from_stage',
        choices=ALL_STAGES,
        metavar='STAGE',
        help=(
            'Run all configured stages starting from STAGE (inclusive): '
            'bib | review | topic-model | topic-report.'
        ),
    )

    args = parser.parse_args(argv)

    pipeline = Pipeline.from_config(args.config)

    if args.from_stage is not None:
        # All configured stages from from_stage onwards (in canonical order)
        start = ALL_STAGES.index(args.from_stage)
        configured = pipeline._configured_stages()
        stages = [s for s in ALL_STAGES[start:] if s in configured]
        if not stages:
            print(f"No configured stages found from '{args.from_stage}' onwards.")
            return
        print(f"Starting pipeline from '{args.from_stage}'…")
        pipeline.run(stages=stages)
        for s in stages:
            _print_stage_result(s, pipeline)
    elif args.stage is not None:
        stages = args.stage
        print(f"Running stage(s): {', '.join(stages)}…")
        pipeline.run(stages=stages)
        for s in stages:
            _print_stage_result(s, pipeline)
    else:
        print("Starting pipeline…")
        pipeline.run()


# =============================================================================
# Download sub-command
# =============================================================================

def _run_download(argv):
    parser = argparse.ArgumentParser(
        prog='pysyrev download',
        description=(
            'Download full-text papers for a list of candidates.\n\n'
            'Sources are tried in cascade order: Unpaywall → OpenAlex → Elsevier TDM.\n'
            'Without --config, only the OpenAlex step runs (no API key required).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'doc_dataset',
        help='CSV file listing the papers to download (must contain a "doi" column).',
    )
    parser.add_argument(
        'output_dir',
        help='Directory where downloaded files and the report will be written.',
    )
    parser.add_argument(
        '--config', '-c',
        metavar='DOWNLOAD_CONFIG',
        default=None,
        help='Path to a download_config.yaml file (provides API keys and options).',
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=None,
        metavar='N',
        help='Cap the number of papers to attempt (useful for testing).',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=None,
        metavar='SECONDS',
        help='Seconds to wait between HTTP requests (default: 1.0).',
    )
    args = parser.parse_args(argv)

    from pysyrev.core.config import DownloadConfig
    from pysyrev.download import PaperDownloader

    if args.config:
        cfg = DownloadConfig.load(args.config)
        # Positional arguments always win over the config file.
        cfg.doc_dataset = args.doc_dataset
        cfg.output_dir  = args.output_dir
        os.makedirs(os.path.join(cfg.output_dir, 'papers'), exist_ok=True)
    else:
        cfg = DownloadConfig(
            doc_dataset=args.doc_dataset,
            output_dir=args.output_dir,
        )

    if args.max_papers is not None:
        cfg.max_papers = args.max_papers
    if args.delay is not None:
        cfg.request_delay = args.delay

    PaperDownloader.from_config(cfg).run().save()


# =============================================================================
# Estimate sub-command
# =============================================================================

def _run_estimate(argv):
    parser = argparse.ArgumentParser(
        prog='pysyrev estimate',
        description=(
            'Estimate the token usage and cost of the review stage before running it.\n\n'
            'The prompts are rebuilt exactly as the review stage sends them and\n'
            'counted with Anthropic\'s count_tokens endpoint (free, no inference).\n'
            'Nothing is reviewed and no model is called.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Path to the YAML pipeline config file.')
    parser.add_argument(
        '--dataset', default=None, metavar='CSV',
        help='Corpus to estimate on (default: review.doc_dataset from the config).',
    )
    parser.add_argument(
        '--escalation-rate', type=float, default=0.10, metavar='RATE',
        help=(
            'Share of documents that reach the next round for lack of consensus '
            '(default: 0.10). Ignored when --calibrate provides a measured value.'
        ),
    )
    parser.add_argument(
        '--calibrate', default=None, metavar='REVIEWED_CSV',
        help=(
            'Measure output length and escalation rate from a previously reviewed '
            'dataset instead of using the built-in defaults.'
        ),
    )
    parser.add_argument(
        '--sample-size', type=int, default=24, metavar='N',
        help='Articles counted exactly to fit the per-article token model (default: 24).',
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help='Also show what other items_per_call settings would cost.',
    )
    parser.add_argument(
        '--offline', action='store_true',
        help='Do not call the API; estimate tokens from character counts (±15 %%).',
    )
    parser.add_argument(
        '--json', action='store_true', dest='as_json',
        help='Emit the estimate as JSON instead of a table.',
    )
    args = parser.parse_args(argv)

    from pysyrev.core.config import Config
    from pysyrev.core.token_cost import (estimate_review, sweep_items_per_call,
                                         calibrate_from_run)

    config = Config.load(args.config)
    if 'review' in config.errors:
        print(f"The `review:` section cannot be read — {config.errors['review']}")
        sys.exit(1)
    if config.review is None:
        print("No `review:` section in the config — nothing to estimate.")
        return

    output_tokens, escalation = None, args.escalation_rate
    if args.calibrate:
        measured = calibrate_from_run(args.calibrate, offline=args.offline)
        output_tokens = measured['output_tokens']
        escalation = measured['escalation_rate']
        if not args.as_json:
            print(f"Calibrated on {args.calibrate}: "
                  f"output {measured['output_tokens']}, "
                  f"escalation {escalation:.1%}")

    estimate = estimate_review(
        config.review,
        dataset         = args.dataset,
        escalation_rate = escalation,
        output_tokens   = output_tokens,
        sample_size     = args.sample_size,
        offline         = args.offline,
    )

    if args.as_json:
        import json
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(estimate.render())
        if args.sweep:
            print(sweep_items_per_call(estimate))


# =============================================================================
# Check sub-command
# =============================================================================

def _render_requirements() -> str:
    """The mandatory sections and keys of a pipeline config."""
    from pysyrev.core.config import (SECTION_CLASSES, STAGE_SECTIONS,
                                     conditional_keys, mandatory_keys)

    lines = ["Sections — every one is optional; a stage only needs its own."]
    for name, cls in SECTION_CLASSES.items():
        keys = mandatory_keys(cls, name)
        lines.append(f"  {name + ':':<15} mandatory keys: "
                     f"{', '.join(keys) if keys else 'none'}")
        for note in conditional_keys(cls, name):
            lines.append(f"  {'':<15} conditional:    {note}")
    lines.append("  Every `<section>.export.export_dir` may be left blank when the "
                 "root-level `output_dir:` is set (→ <output_dir>/<section>/).")

    lines += ["", "Stages — `--stage <name>` only reads the sections below."]
    for stage, spec in STAGE_SECTIONS.items():
        uses = f"  (uses {', '.join(spec['uses'])} when valid)" if spec['uses'] else ""
        lines.append(f"  {stage:<15} needs {', '.join(spec['needs'])}{uses}")
    return "\n".join(lines)


def _run_check(argv):
    parser = argparse.ArgumentParser(
        prog='pysyrev check',
        description=(
            'Report which config sections and keys are mandatory, and — when a\n'
            'config file is given — which of its stages are ready to run.\n'
            'Nothing is executed and no API is called.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'config', nargs='?', default=None,
        help='Pipeline config to check. Omit to only list what is mandatory.',
    )
    parser.add_argument(
        '--stage', nargs='+', choices=ALL_STAGES, metavar='STAGE', default=None,
        help='Only check these stages (default: every stage the config declares).',
    )
    args = parser.parse_args(argv)

    print(_render_requirements())
    if args.config is None:
        return

    from pysyrev.core.config import Config

    print(f"\nChecking {args.config}")
    config = Config.load(args.config)
    stages = args.stage or [s for s in ALL_STAGES if config.stage_is_declared(s)]
    if not stages:
        print("  no stage is declared — the config has no runnable section.")
        return

    ready = True
    for stage in stages:
        try:
            config.require_stages([stage])
            print(f"  {stage:<15} ok")
        except ValueError as exc:
            ready = False
            print(f"  {stage:<15} NOT RUNNABLE — {exc}")

    if not ready:
        sys.exit(1)


# =============================================================================
# Serve sub-command
# =============================================================================

def _run_serve(argv):
    parser = argparse.ArgumentParser(
        prog='pysyrev serve',
        description=(
            'Serve the topic report as a live page: the interactive HTML report\n'
            'plus a panel of settings. Changing one rebuilds its section in place;\n'
            'the panel shows the changed settings as YAML to paste back into the\n'
            'config. Reads the latest topic-model run; writes nothing.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Path to the YAML pipeline config file.')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Address to listen on (default: 127.0.0.1, this machine only).')
    parser.add_argument('--port', type=int, default=8765,
                        help='Port to listen on (default: 8765; a free one if taken).')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open the page in the browser.')
    args = parser.parse_args(argv)

    from pysyrev.core.config import Config
    from pysyrev.topic_report import TopicReport

    config = Config.load(args.config)
    config.require_stages(['topic-report'])
    print("Building the report…")
    TopicReport.from_config(config).serve(
        host=args.host, port=args.port, open_browser=not args.no_browser)


# =============================================================================
# Entry point
# =============================================================================

def main():
    # Route to a sub-command when the first argument names one, preserving full
    # backward compatibility with `pysyrev config.yaml`.
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        _run_download(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'estimate':
        _run_estimate(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'check':
        _run_check(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'serve':
        _run_serve(sys.argv[2:])
    else:
        _run_pipeline(sys.argv[1:])


if __name__ == '__main__':
    main()
