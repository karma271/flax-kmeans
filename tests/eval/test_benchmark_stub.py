"""Tests for Phase C benchmark stub helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.eval.benchmark_stub import _detect_device, run_benchmark_stub, write_benchmark_artifacts
from src.eval.contracts import ExperimentConfig, MetricRecord, RunManifest


def _build_dataset() -> np.ndarray:
    x, _ = make_blobs(
        n_samples=120,
        centers=[(-6.0, -6.0), (0.0, 5.0), (6.0, -1.0)],
        cluster_std=0.35,
        random_state=13,
    )
    return np.asarray(x, dtype=np.float32)


def test_run_benchmark_stub_for_sklearn_outputs_contracts() -> None:
    """Runner returns manifest and metrics records for sklearn implementation."""
    config = ExperimentConfig(
        run_id="test_run_sklearn_stub",
        implementation="sklearn_kmeans",
        dataset_id="synthetic_stub",
        random_seed=3,
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        n_init=4,
    )
    x = _build_dataset()
    manifest, metrics = run_benchmark_stub(config, x)

    assert manifest.run_id == config.run_id
    assert manifest.implementation == "sklearn_kmeans"
    assert manifest.device == "cpu"
    assert manifest.dataset_id == config.dataset_id
    assert manifest.n_samples == x.shape[0]
    assert manifest.n_features == x.shape[1]
    assert manifest.n_clusters == config.n_clusters
    assert manifest.fit_time_ms >= 0.0
    assert manifest.predict_time_ms >= 0.0
    assert manifest.peak_memory_mb is None or manifest.peak_memory_mb >= 0.0
    assert manifest.iterations_used > 0
    assert isinstance(manifest.converged, bool)
    assert isinstance(manifest.software_versions, dict)

    assert metrics.run_id == config.run_id
    assert metrics.implementation == config.implementation
    assert metrics.dataset_id == config.dataset_id
    assert np.isfinite(metrics.inertia)
    assert metrics.silhouette is None or -1.0 <= metrics.silhouette <= 1.0
    assert metrics.calinski_harabasz is None or metrics.calinski_harabasz >= 0.0
    assert metrics.davies_bouldin is None or metrics.davies_bouldin >= 0.0


def test_write_benchmark_artifacts_persists_json(tmp_path: Path) -> None:
    """Writer stores both records under run_id filenames."""
    config = ExperimentConfig(
        run_id="test_run_write_stub",
        implementation="sklearn_kmeans",
        dataset_id="synthetic_stub_write",
        random_seed=4,
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        n_init=3,
    )
    x = _build_dataset()
    manifest, metrics = run_benchmark_stub(config, x)
    manifest_path, metrics_path = write_benchmark_artifacts(
        manifest,
        metrics,
        manifests_dir=tmp_path / "manifests",
        metrics_dir=tmp_path / "metrics",
    )

    assert manifest_path.exists()
    assert metrics_path.exists()

    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert manifest_json["run_id"] == config.run_id
    assert metrics_json["run_id"] == config.run_id


def test_detect_device_defaults_to_cpu_for_jax_cpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """JAX runtimes without GPU/TPU are reported as CPU."""

    class _FakeJax:
        @staticmethod
        def default_backend() -> str:
            return "cpu"

    monkeypatch.setitem(sys.modules, "jax", _FakeJax())
    assert _detect_device("jax_kmeans") == "cpu"


def test_write_benchmark_artifacts_rejects_invalid_metrics(tmp_path: Path) -> None:
    """Artifact writer validates payload schema before persisting files."""
    manifest = RunManifest(
        run_id="invalid_metrics_run",
        timestamp_utc=RunManifest.utc_now(),
        implementation="sklearn_kmeans",
        device="cpu",
        dataset_id="synthetic_stub",
        n_samples=100,
        n_features=2,
        n_clusters=3,
        random_seed=7,
        max_iter=100,
        tolerance=1e-4,
        n_init=4,
        converged=True,
        iterations_used=10,
        fit_time_ms=1.0,
        predict_time_ms=0.0,
        peak_memory_mb=None,
        software_versions={"numpy": "0.0"},
    )
    metrics = MetricRecord(
        run_id="invalid_metrics_run",
        implementation="sklearn_kmeans",
        dataset_id="synthetic_stub",
        inertia=12.5,
        silhouette=2.0,
        calinski_harabasz=4.0,
        davies_bouldin=0.2,
    )

    with pytest.raises(ValueError, match="MetricRecord schema validation failed"):
        write_benchmark_artifacts(
            manifest,
            metrics,
            manifests_dir=tmp_path / "manifests",
            metrics_dir=tmp_path / "metrics",
        )
