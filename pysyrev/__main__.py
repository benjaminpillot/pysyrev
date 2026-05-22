"""
pysyrev CLI — run the literature review pipeline from a YAML config.

Usage
-----
  # Run all configured stages (only stages present in the config are executed)
  python -m pysyrev config.yaml

  # Individual stages
  python -m pysyrev config.yaml --stage bib
  python -m pysyrev config.yaml --stage review
  python -m pysyrev config.yaml --stage bib-network
  python -m pysyrev config.yaml --stage topic-model
  python -m pysyrev config.yaml --stage topic-report

When running a single stage, the input dataset is read from the
``doc_dataset`` field of that stage's config section.  If ``doc_dataset``
is left blank, Config.load falls back to auto-detecting the most recent
output of the previous stage (e.g. the latest ``reviewed_included.csv``
for bib-network / topic-model, or the latest topic_model run for
topic-report).
"""

import argparse

from pysyrev import ALL_STAGES, Pipeline


def _print_stage_result(stage, pipeline):
    if stage == 'bib' and pipeline.bib is not None:
        print(f"[bib] Done — {len(pipeline.bib.dataset)} documents.")
    elif stage == 'review' and pipeline.review is not None:
        print(f"[review] Done — {len(pipeline.review.included_docs)} documents included.")
    elif stage == 'bib-network' and pipeline.network is not None:
        net = pipeline.network
        if pipeline.config.bib_network.export is not None:
            print(
                f"[bib-network] Done — coupling: {net.n_coupling_nodes} nodes / "
                f"{net.n_coupling_edges} edges, "
                f"co-citation: {net.n_cocitation_nodes} nodes / "
                f"{net.n_cocitation_edges} edges."
            )
        else:
            print("[bib-network] Done (no export configured).")
    elif stage == 'topic-model':
        print("[topic-model] Done.")
    elif stage == 'topic-report' and pipeline.report is not None:
        print(f"[topic-report] Done — report written to {pipeline.report.export_to}")


def main():
    parser = argparse.ArgumentParser(
        prog='python -m pysyrev',
        description='Systematic literature review pipeline.',
    )
    parser.add_argument('config', help='Path to the YAML pipeline config file.')
    parser.add_argument(
        '--stage',
        choices=ALL_STAGES + ['all'],
        default='all',
        metavar='STAGE',
        help=(
            'Stage to run: bib | review | bib-network | topic-model | topic-report | all '
            '(default: all — runs only the stages present in the config file).'
        ),
    )
    args = parser.parse_args()

    pipeline = Pipeline.from_config(args.config)

    if args.stage == 'all':
        print("Starting pipeline…")
        pipeline.run()
    else:
        print(f"[{args.stage}] Starting…")
        pipeline.run(stages=[args.stage])
        _print_stage_result(args.stage, pipeline)


if __name__ == '__main__':
    main()
