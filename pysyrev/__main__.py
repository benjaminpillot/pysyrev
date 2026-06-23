"""
pysyrev CLI — systematic literature review pipeline.

Usage
-----
  # Run all configured pipeline stages
  pysyrev config.yaml

  # Run one or more specific stages
  pysyrev config.yaml --stage bib-network
  pysyrev config.yaml --stage bib-network topic-model topic-report

  # Run from a given stage to the end (all configured stages from that point)
  pysyrev config.yaml --from bib-network

  # Download full-text papers for a list of candidates
  pysyrev download liste.csv output_folder [--config download_config.yaml]

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
    elif stage == 'bib-network' and pipeline.network is not None:
        net = pipeline.network
        if pipeline.config.bib_network.export is not None:
            print(
                f"[bib-network] Done — "
                f"citation: {net.n_citation_nodes} nodes / {net.n_citation_edges} edges, "
                f"coupling: {net.n_coupling_nodes} nodes / {net.n_coupling_edges} edges, "
                f"co-citation: {net.n_cocitation_nodes} nodes / {net.n_cocitation_edges} edges."
            )
        else:
            print("[bib-network] Done (no export configured).")
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
            'bib | review | bib-network | topic-model | topic-report. '
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
            'bib | review | bib-network | topic-model | topic-report.'
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
# Entry point
# =============================================================================

def main():
    # Route to the download sub-command when the first argument is "download",
    # preserving full backward compatibility with `pysyrev config.yaml`.
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        _run_download(sys.argv[2:])
    else:
        _run_pipeline(sys.argv[1:])


if __name__ == '__main__':
    main()
