"""Notebook-friendly orchestration helpers for cross-device benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.data.generate_synthetic_npy import generate_synthetic_blobs
from src.eval.benchmark_stub import run_benchmark_stub, write_benchmark_artifacts
from src.eval.contracts import ExperimentConfig, Implementation

CANONICAL_IMPLEMENTATIONS: tuple[Implementation, ...] = (
    "jax_kmeans",
    "jax_flash_kmeans",
    "sklearn_kmeans",
    "flashkmeans_wrapper",
)


@dataclass(slots=True)
class DatasetSpec:
    """Synthetic dataset shape and generation parameters."""

    dataset_id: str
    n_samples: int
    n_features: int
    n_clusters: int
    cluster_std: float

    @property
    def scale_id(self) -> str:
        """Scale identifier matching comparative analysis conventions."""
        return f"{self.n_samples}x{self.n_features}_k{self.n_clusters}"


@dataclass(slots=True)
class BenchmarkMatrix:
    """Shared benchmark definition used by all execution notebooks."""

    exp_name: str
    seeds: list[int]
    max_iter: int
    tolerance: float
    n_init: int
    datasets: list[DatasetSpec]
    implementation_kwargs: dict[str, dict[str, Any]]


def _as_str_dict(value: Any) -> dict[str, Any]:
    """Validate and cast arbitrary YAML object into a string-key dictionary."""
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping/object in benchmark matrix.")
    return {str(k): v for k, v in value.items()}


def load_benchmark_matrix(path: str | Path) -> BenchmarkMatrix:
    """Load benchmark matrix YAML file for notebook execution."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _as_str_dict(payload)

    exp_name = str(root["exp_name"])
    seeds_raw = root["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("Benchmark matrix 'seeds' must be a non-empty list.")
    seeds = [int(seed) for seed in seeds_raw]

    run_config = _as_str_dict(root["run_config"])
    max_iter = int(run_config["max_iter"])
    tolerance = float(run_config["tolerance"])
    n_init = int(run_config["n_init"])

    datasets_raw = root["datasets"]
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValueError("Benchmark matrix 'datasets' must be a non-empty list.")
    datasets: list[DatasetSpec] = []
    for dataset_obj in datasets_raw:
        dataset = _as_str_dict(dataset_obj)
        datasets.append(
            DatasetSpec(
                dataset_id=str(dataset["dataset_id"]),
                n_samples=int(dataset["n_samples"]),
                n_features=int(dataset["n_features"]),
                n_clusters=int(dataset["n_clusters"]),
                cluster_std=float(dataset["cluster_std"]),
            )
        )

    implementation_kwargs_raw = root.get("implementation_kwargs", {})
    implementation_kwargs_obj = _as_str_dict(implementation_kwargs_raw)
    implementation_kwargs = {
        key: _as_str_dict(value) for key, value in implementation_kwargs_obj.items()
    }

    return BenchmarkMatrix(
        exp_name=exp_name,
        seeds=seeds,
        max_iter=max_iter,
        tolerance=tolerance,
        n_init=n_init,
        datasets=datasets,
        implementation_kwargs=implementation_kwargs,
    )


def build_run_id(
    *,
    exp_name: str,
    implementation: str,
    device: str,
    dataset_id: str,
    scale_id: str,
    seed: int,
) -> str:
    """Build deterministic run IDs for multi-session Colab executions."""
    return (
        f"{exp_name}__{implementation}__{device}__{dataset_id}__{scale_id}__"
        f"s{seed:04d}"
    )


def run_matrix_for_implementation(
    *,
    matrix_path: str | Path,
    implementation: Implementation,
    target_device: str,
    results_root: str | Path,
    overwrite_existing: bool = False,
) -> list[dict[str, str]]:
    """Run all matrix combinations for one implementation and persist artifacts."""
    if implementation not in CANONICAL_IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {implementation}")

    matrix = load_benchmark_matrix(matrix_path)
    results_dir = Path(results_root)
    manifests_dir = results_dir / "manifests"
    metrics_dir = results_dir / "metrics"

    implementation_kwargs = matrix.implementation_kwargs.get(implementation, {})
    records: list[dict[str, str]] = []

    for dataset in matrix.datasets:
        for seed in matrix.seeds:
            run_id = build_run_id(
                exp_name=matrix.exp_name,
                implementation=implementation,
                device=target_device,
                dataset_id=dataset.dataset_id,
                scale_id=dataset.scale_id,
                seed=seed,
            )
            manifest_path = manifests_dir / f"{run_id}.json"
            metric_path = metrics_dir / f"{run_id}.json"

            if (
                not overwrite_existing
                and manifest_path.exists()
                and metric_path.exists()
            ):
                records.append(
                    {
                        "run_id": run_id,
                        "manifest_path": str(manifest_path),
                        "metric_path": str(metric_path),
                        "status": "skipped_existing",
                    }
                )
                continue

            x, _ = generate_synthetic_blobs(
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                n_clusters=dataset.n_clusters,
                cluster_std=dataset.cluster_std,
                random_seed=seed,
            )
            config = ExperimentConfig(
                run_id=run_id,
                implementation=implementation,
                dataset_id=dataset.dataset_id,
                random_seed=seed,
                n_clusters=dataset.n_clusters,
                max_iter=matrix.max_iter,
                tolerance=matrix.tolerance,
                n_init=matrix.n_init,
                notes=f"{matrix.exp_name} notebook benchmark run",
            )
            manifest, metrics = run_benchmark_stub(
                config=config,
                x=x,
                implementation_kwargs=implementation_kwargs,
            )
            written_manifest, written_metric = write_benchmark_artifacts(
                manifest=manifest,
                metrics=metrics,
                manifests_dir=manifests_dir,
                metrics_dir=metrics_dir,
            )
            records.append(
                {
                    "run_id": run_id,
                    "manifest_path": str(written_manifest),
                    "metric_path": str(written_metric),
                    "status": "written",
                }
            )

    return records


def expected_run_ids(
    *,
    matrix_path: str | Path,
    implementation_device_map: dict[Implementation, str],
) -> list[str]:
    """Return deterministic expected run IDs for all implementation/device pairs."""
    matrix = load_benchmark_matrix(matrix_path)
    run_ids: list[str] = []
    for implementation, device in implementation_device_map.items():
        for dataset in matrix.datasets:
            for seed in matrix.seeds:
                run_ids.append(
                    build_run_id(
                        exp_name=matrix.exp_name,
                        implementation=implementation,
                        device=device,
                        dataset_id=dataset.dataset_id,
                        scale_id=dataset.scale_id,
                        seed=seed,
                    )
                )
    return sorted(run_ids)


def existing_run_ids(results_root: str | Path) -> set[str]:
    """Collect run IDs that already have both manifest and metric artifacts."""
    results_dir = Path(results_root)
    manifests_dir = results_dir / "manifests"
    metrics_dir = results_dir / "metrics"
    manifest_ids = {path.stem for path in manifests_dir.glob("*.json")}
    metric_ids = {path.stem for path in metrics_dir.glob("*.json")}
    return manifest_ids & metric_ids
