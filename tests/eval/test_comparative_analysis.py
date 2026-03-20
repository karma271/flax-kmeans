"""Tests for comparative analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.comparative_analysis import (
    build_comparative_dataframe,
    build_comparative_figures,
    build_summary_tables,
    load_artifact_frames,
    write_analysis_outputs,
)
from src.eval.run_comparative_analysis import run_comparative_analysis


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_artifacts(root: Path) -> None:
    manifests_dir = root / "manifests"
    metrics_dir = root / "metrics"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = [
        {
            "run_id": "r1",
            "timestamp_utc": "2026-03-19T01:00:00+00:00",
            "implementation": "jax_kmeans",
            "device": "cpu",
            "dataset_id": "synthetic_blobs",
            "n_samples": 200,
            "n_features": 8,
            "n_clusters": 4,
            "random_seed": 11,
            "max_iter": 100,
            "tolerance": 1e-4,
            "n_init": 4,
            "converged": True,
            "iterations_used": 13,
            "fit_time_ms": 10.0,
            "predict_time_ms": 3.0,
            "peak_memory_mb": 40.0,
        },
        {
            "run_id": "r2",
            "timestamp_utc": "2026-03-19T01:02:00+00:00",
            "implementation": "sklearn_kmeans",
            "device": "cpu",
            "dataset_id": "synthetic_blobs",
            "n_samples": 200,
            "n_features": 8,
            "n_clusters": 4,
            "random_seed": 11,
            "max_iter": 100,
            "tolerance": 1e-4,
            "n_init": 4,
            "converged": True,
            "iterations_used": 12,
            "fit_time_ms": 8.0,
            "predict_time_ms": 2.0,
            "peak_memory_mb": 38.0,
        },
        {
            "run_id": "r3",
            "timestamp_utc": "2026-03-19T01:05:00+00:00",
            "implementation": "jax_kmeans",
            "device": "gpu",
            "dataset_id": "synthetic_blobs",
            "n_samples": 1000,
            "n_features": 16,
            "n_clusters": 8,
            "random_seed": 15,
            "max_iter": 200,
            "tolerance": 1e-4,
            "n_init": 5,
            "converged": True,
            "iterations_used": 24,
            "fit_time_ms": 18.0,
            "predict_time_ms": 5.0,
            "peak_memory_mb": 120.0,
        },
    ]
    metric_records = [
        {
            "run_id": "r1",
            "implementation": "jax_kmeans",
            "dataset_id": "synthetic_blobs",
            "inertia": 90.0,
            "silhouette": 0.62,
            "calinski_harabasz": 460.0,
            "davies_bouldin": 0.55,
        },
        {
            "run_id": "r2",
            "implementation": "sklearn_kmeans",
            "dataset_id": "synthetic_blobs",
            "inertia": 92.0,
            "silhouette": 0.64,
            "calinski_harabasz": 455.0,
            "davies_bouldin": 0.52,
        },
        {
            "run_id": "r3",
            "implementation": "jax_kmeans",
            "dataset_id": "synthetic_blobs",
            "inertia": 400.0,
            "silhouette": 0.48,
            "calinski_harabasz": 800.0,
            "davies_bouldin": 0.83,
        },
    ]
    for record in manifest_records:
        _write_json(manifests_dir / f"{record['run_id']}.json", record)
    for record in metric_records:
        _write_json(metrics_dir / f"{record['run_id']}.json", record)


def test_build_comparative_dataframe_merges_and_derives_fields(tmp_path: Path) -> None:
    """Comparative frame includes merged contract fields and scale metadata."""
    _seed_artifacts(tmp_path)
    manifests, metrics = load_artifact_frames(tmp_path)
    comparative = build_comparative_dataframe(manifests, metrics)

    assert len(comparative) == 3
    assert {"total_time_ms", "scale_id", "implementation", "dataset_id"} <= set(comparative.columns)
    assert comparative["timestamp_utc"].dtype.name.startswith("datetime64")
    assert pytest.approx(comparative.loc[0, "total_time_ms"]) == (
        comparative.loc[0, "fit_time_ms"] + comparative.loc[0, "predict_time_ms"]
    )


def test_build_summary_tables_shapes_and_run_counts(tmp_path: Path) -> None:
    """Summary tables include expected grouping and run-count columns."""
    _seed_artifacts(tmp_path)
    manifests, metrics = load_artifact_frames(tmp_path)
    comparative = build_comparative_dataframe(manifests, metrics)
    tables = build_summary_tables(comparative)

    assert set(tables.keys()) == {
        "comparative_runs",
        "summary_by_device_scale",
        "summary_by_implementation",
    }
    assert "run_count" in tables["summary_by_device_scale"].columns
    assert int(tables["summary_by_device_scale"]["run_count"].min()) >= 1
    assert "total_time_ms_median" in tables["summary_by_implementation"].columns
    assert int(tables["summary_by_implementation"]["run_count"].sum()) == len(comparative)


def test_write_analysis_outputs_persists_csv_html_and_report(tmp_path: Path) -> None:
    """Output writer emits CSV tables, plot HTML files, and markdown report."""
    _seed_artifacts(tmp_path)
    manifests, metrics = load_artifact_frames(tmp_path)
    comparative = build_comparative_dataframe(manifests, metrics)
    tables = build_summary_tables(comparative)
    figures = build_comparative_figures(comparative)
    outputs = write_analysis_outputs(tables, figures, output_root=tmp_path / "outputs")

    assert (tmp_path / "outputs/metrics/comparative_runs.csv").exists()
    assert (tmp_path / "outputs/metrics/summary_by_device_scale.csv").exists()
    assert (tmp_path / "outputs/plots/tradeoff_speed_vs_quality.html").exists()
    assert (tmp_path / "outputs/plots/tradeoff_report.md").exists()
    assert "tradeoff_report" in outputs


def test_run_comparative_analysis_end_to_end(tmp_path: Path) -> None:
    """CLI wrapper runs the full analysis pipeline and writes outputs."""
    _seed_artifacts(tmp_path / "results")
    outputs = run_comparative_analysis(tmp_path / "results", tmp_path / "results")
    comparative_csv = outputs["comparative_runs"]
    frame = pd.read_csv(comparative_csv)
    assert not frame.empty
