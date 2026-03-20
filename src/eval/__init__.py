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

__all__ = [
    "ExperimentConfig",
    "MetricRecord",
    "RunManifest",
    "build_comparative_dataframe",
    "build_comparative_figures",
    "build_summary_tables",
    "load_artifact_frames",
    "load_experiment_config",
    "run_benchmark_stub",
    "write_analysis_outputs",
    "write_benchmark_artifacts",
]
