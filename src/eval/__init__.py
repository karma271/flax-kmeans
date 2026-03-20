"""Benchmarking and clustering evaluation utilities."""

from src.eval.benchmark_stub import run_benchmark_stub, write_benchmark_artifacts
from src.eval.comparative_analysis import (
    build_comparative_dataframe,
    build_comparative_figures,
    build_summary_tables,
    load_artifact_frames,
    write_analysis_outputs,
)
from src.eval.config_io import load_experiment_config
from src.eval.contracts import ExperimentConfig, MetricRecord, RunManifest
from src.eval.notebook_harness import (
    existing_run_ids,
    expected_run_ids,
    load_benchmark_matrix,
    run_matrix_for_implementation,
)

__all__ = [
    "ExperimentConfig",
    "MetricRecord",
    "RunManifest",
    "build_comparative_dataframe",
    "build_comparative_figures",
    "build_summary_tables",
    "load_artifact_frames",
    "load_benchmark_matrix",
    "load_experiment_config",
    "run_benchmark_stub",
    "run_matrix_for_implementation",
    "expected_run_ids",
    "existing_run_ids",
    "write_analysis_outputs",
    "write_benchmark_artifacts",
]
