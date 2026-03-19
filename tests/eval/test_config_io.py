"""Tests for experiment config YAML loading."""

from __future__ import annotations

from pathlib import Path

from src.eval.config_io import load_experiment_config


def test_load_experiment_config_reads_yaml_contract(tmp_path: Path) -> None:
    """YAML payload is parsed into typed ExperimentConfig."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                'run_id: "exp_test_001"',
                'implementation: "sklearn_kmeans"',
                'dataset_id: "synthetic_blobs_s1"',
                "random_seed: 42",
                "n_clusters: 3",
                "max_iter: 200",
                "tolerance: 0.0001",
                "n_init: 5",
                "batch_size: null",
                'notes: "test config"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.run_id == "exp_test_001"
    assert config.implementation == "sklearn_kmeans"
    assert config.n_clusters == 3
    assert config.n_init == 5
