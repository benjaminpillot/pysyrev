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

  # Price the review stage before paying for it (no model call)
  pysyrev estimate config.yaml [--calibrate reviewed.csv] [--sweep]

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
# Entry point
# =============================================================================

def main():
    # Route to a sub-command when the first argument names one, preserving full
    # backward compatibility with `pysyrev config.yaml`.
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        _run_download(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'estimate':
        _run_estimate(sys.argv[2:])
    else:
        _run_pipeline(sys.argv[1:])


if __name__ == '__main__':
    main()
