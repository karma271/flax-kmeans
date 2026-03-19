"""Tests for lightweight benchmark CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

from src.eval.run_benchmark import run_from_args


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'run_id: "exp_cli_001"',
                'implementation: "sklearn_kmeans"',
                'dataset_id: "synthetic_cli"',
                "random_seed: 42",
                "n_clusters: 3",
                "max_iter: 120",
                "tolerance: 0.0001",
                "n_init: 4",
            ]
        ),
        encoding="utf-8",
    )


def _write_x(path: Path) -> None:
    x, _ = make_blobs(
        n_samples=100,
        centers=[(-6.0, -6.0), (0.0, 5.0), (6.0, -1.0)],
        cluster_std=0.4,
        random_state=7,
    )
    np.save(path, np.asarray(x, dtype=np.float32))


def test_run_from_args_writes_artifacts(tmp_path: Path) -> None:
    """CLI helper writes manifest/metrics files into provided root."""
    config_path = tmp_path / "config.yaml"
    x_path = tmp_path / "x.npy"
    results_root = tmp_path / "results"
    _write_config(config_path)
    _write_x(x_path)

    manifest_path, metrics_path = run_from_args(
        [
            "--config",
            str(config_path),
            "--x-npy",
            str(x_path),
            "--results-root",
            str(results_root),
        ]
    )

    assert manifest_path is not None
    assert metrics_path is not None
    assert manifest_path.exists()
    assert metrics_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert manifest_payload["run_id"] == "exp_cli_001"
    assert metrics_payload["run_id"] == "exp_cli_001"


def test_run_from_args_no_write_mode(tmp_path: Path) -> None:
    """CLI helper can skip artifact writing when requested."""
    config_path = tmp_path / "config.yaml"
    x_path = tmp_path / "x.npy"
    _write_config(config_path)
    _write_x(x_path)

    manifest_path, metrics_path = run_from_args(
        [
            "--config",
            str(config_path),
            "--x-npy",
            str(x_path),
            "--no-write",
        ]
    )
    assert manifest_path is None
    assert metrics_path is None
