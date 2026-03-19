"""Tests for Phase C benchmark stub helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

from src.eval.benchmark_stub import run_benchmark_stub, write_benchmark_artifacts
from src.eval.contracts import ExperimentConfig


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
    assert manifest.predict_time_ms == 0.0
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
