"""CLI for comparative analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.comparative_analysis import (
    build_comparative_dataframe,
    build_comparative_figures,
    build_summary_tables,
    load_artifact_frames,
    write_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build parser for comparative analysis entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run comparative analysis over benchmark artifacts."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Location of benchmark artifacts (must contain manifests/ and metrics/).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Destination for comparative outputs (defaults to --results-root).",
    )
    return parser


def run_comparative_analysis(results_root: Path, output_root: Path) -> dict[str, Path]:
    """Run comparative analysis and write tables/plots outputs."""
    manifests, metrics = load_artifact_frames(results_root)
    comparative = build_comparative_dataframe(manifests, metrics)
    summary_tables = build_summary_tables(comparative)
    figures = build_comparative_figures(comparative)
    return write_analysis_outputs(summary_tables, figures, output_root=output_root)


def run_from_args(argv: list[str] | None = None) -> dict[str, Path]:
    """Execute the CLI workflow from command line style args."""
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root
    return run_comparative_analysis(results_root, output_root)


def main() -> None:
    """Entrypoint for python -m execution."""
    run_from_args()


if __name__ == "__main__":
    main()
