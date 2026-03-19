"""CLI entrypoint for running one benchmark config against one dataset array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.benchmark_stub import run_benchmark_stub, write_benchmark_artifacts
from src.eval.config_io import load_experiment_config


def _parse_kwargs(raw: str | None) -> dict[str, Any]:
    """Parse optional JSON implementation kwargs."""
    if raw is None:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--implementation-kwargs must be a JSON object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for benchmark stub execution."""
    parser = argparse.ArgumentParser(description="Run one flax-kmeans benchmark config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config YAML file.",
    )
    parser.add_argument(
        "--x-npy",
        required=True,
        help="Path to input feature array stored as .npy (shape: n_samples, n_features).",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Base folder for outputs (manifests/, metrics/).",
    )
    parser.add_argument(
        "--implementation-kwargs",
        default=None,
        help="Optional JSON object of implementation-specific kwargs.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute outputs without writing JSON artifact files.",
    )
    return parser


def run_from_args(argv: list[str] | None = None) -> tuple[Path | None, Path | None]:
    """Run benchmark workflow from command-line style arguments."""
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    x = np.load(args.x_npy)
    implementation_kwargs = _parse_kwargs(args.implementation_kwargs)

    manifest, metrics = run_benchmark_stub(
        config=config,
        x=x,
        implementation_kwargs=implementation_kwargs,
    )

    if args.no_write:
        return None, None

    results_root = Path(args.results_root)
    manifests_dir = results_root / "manifests"
    metrics_dir = results_root / "metrics"
    return write_benchmark_artifacts(
        manifest=manifest,
        metrics=metrics,
        manifests_dir=manifests_dir,
        metrics_dir=metrics_dir,
    )


def main() -> None:
    """Entrypoint for python -m invocation."""
    run_from_args()


if __name__ == "__main__":
    main()
