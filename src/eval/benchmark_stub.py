"""Minimal benchmark runner stub for Phase C and Phase D integration."""

from __future__ import annotations

import importlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.algorithms.result import KMeansFitResult
from src.eval.contracts import Device, ExperimentConfig, MetricRecord, RunManifest
from src.eval.schema_validation import validate_metric_record_payload, validate_run_manifest_payload

Runner = Callable[..., KMeansFitResult]


def _resolve_runner(implementation: str) -> Runner:
    """Return fit function for the canonical implementation id."""
    if implementation == "jax_kmeans":
        from src.algorithms.jax_kmeans import fit_jax_kmeans

        return fit_jax_kmeans
    if implementation == "jax_flash_kmeans":
        from src.algorithms.jax_flash_kmeans import fit_jax_flash_kmeans

        return fit_jax_flash_kmeans
    if implementation == "sklearn_kmeans":
        from src.algorithms.sklearn_kmeans import fit_sklearn_kmeans

        return fit_sklearn_kmeans
    if implementation == "flashkmeans_wrapper":
        from src.algorithms.flashkmeans_wrapper import fit_flashkmeans_wrapper

        return fit_flashkmeans_wrapper
    raise ValueError(f"Unsupported implementation: {implementation!r}")


def _detect_device(implementation: str) -> Device:
    """Infer device used by current runtime."""
    if implementation == "sklearn_kmeans":
        return "cpu"

    if implementation == "flashkmeans_wrapper":
        try:
            torch = importlib.import_module("torch")

            return "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    try:
        import jax

        backend = jax.default_backend()
    except Exception:
        return "cpu"

    if backend == "tpu":
        return "tpu"
    if backend in {"gpu", "cuda"}:
        return "gpu"
    return "cpu"


def _software_versions(implementation: str) -> dict[str, str]:
    """Collect version metadata for manifests."""
    versions: dict[str, str] = {"numpy": np.__version__}

    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass

    if implementation in {"jax_kmeans", "jax_flash_kmeans"}:
        try:
            import jax

            versions["jax"] = jax.__version__
        except Exception:
            pass

    if implementation == "flashkmeans_wrapper":
        try:
            flash_kmeans = importlib.import_module("flash_kmeans")

            versions["flash-kmeans"] = getattr(flash_kmeans, "__version__", "unknown")
        except Exception:
            pass
        try:
            torch = importlib.import_module("torch")

            versions["torch"] = torch.__version__
        except Exception:
            pass

    return versions


def _compute_quality_metrics(
    x: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Compute optional clustering quality metrics safely."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= x.shape[0]:
        return None, None, None

    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    try:
        silhouette = float(silhouette_score(x, labels))
    except ValueError:
        silhouette = None
    try:
        calinski_harabasz = float(calinski_harabasz_score(x, labels))
    except ValueError:
        calinski_harabasz = None
    try:
        davies_bouldin = float(davies_bouldin_score(x, labels))
    except ValueError:
        davies_bouldin = None

    return silhouette, calinski_harabasz, davies_bouldin


def _measure_predict_time_ms(
    x: np.ndarray,
    centroids: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> float:
    """Measure nearest-centroid assignment time in milliseconds."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    if x.shape[1] != centroids.shape[1]:
        raise ValueError("x and centroids must have the same number of features.")

    start = time.perf_counter()
    for start_idx in range(0, x.shape[0], chunk_size):
        stop_idx = min(start_idx + chunk_size, x.shape[0])
        x_chunk = x[start_idx:stop_idx]
        dists = np.sum((x_chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        _ = np.argmin(dists, axis=1)
    return (time.perf_counter() - start) * 1000.0


def _peak_memory_mb() -> float | None:
    """Return peak resident memory for current process when available."""
    try:
        import resource

        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if max_rss <= 0.0:
        return None
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def run_benchmark_stub(
    config: ExperimentConfig,
    x: np.ndarray,
    *,
    implementation_kwargs: dict[str, Any] | None = None,
) -> tuple[RunManifest, MetricRecord]:
    """Run one config against one implementation and return benchmark artifacts."""
    x_array = np.asarray(x, dtype=np.float32)
    if x_array.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features).")

    fit_fn = _resolve_runner(config.implementation)
    kwargs = implementation_kwargs or {}

    start = time.perf_counter()
    result = fit_fn(
        x=x_array,
        n_clusters=config.n_clusters,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        n_init=config.n_init,
        random_seed=config.random_seed,
        **kwargs,
    )
    fit_time_ms = (time.perf_counter() - start) * 1000.0

    centroids = np.asarray(result.centroids, dtype=np.float32)
    predict_time_ms = _measure_predict_time_ms(
        x=x_array,
        centroids=centroids,
    )

    silhouette, calinski_harabasz, davies_bouldin = _compute_quality_metrics(
        x=x_array,
        labels=result.labels,
    )
    manifest = RunManifest(
        run_id=config.run_id,
        timestamp_utc=RunManifest.utc_now(),
        implementation=config.implementation,
        device=_detect_device(config.implementation),
        dataset_id=config.dataset_id,
        n_samples=int(x_array.shape[0]),
        n_features=int(x_array.shape[1]),
        n_clusters=config.n_clusters,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        n_init=config.n_init,
        converged=result.converged,
        iterations_used=result.n_iter,
        fit_time_ms=float(fit_time_ms),
        predict_time_ms=float(predict_time_ms),
        peak_memory_mb=_peak_memory_mb(),
        software_versions=_software_versions(config.implementation),
    )
    metrics = MetricRecord(
        run_id=config.run_id,
        implementation=config.implementation,
        dataset_id=config.dataset_id,
        inertia=result.inertia,
        silhouette=silhouette,
        calinski_harabasz=calinski_harabasz,
        davies_bouldin=davies_bouldin,
    )
    return manifest, metrics


def write_benchmark_artifacts(
    manifest: RunManifest,
    metrics: MetricRecord,
    *,
    manifests_dir: Path = Path("results/manifests"),
    metrics_dir: Path = Path("results/metrics"),
) -> tuple[Path, Path]:
    """Write manifest and metrics records as JSON files by run id."""
    manifest_payload = manifest.to_dict()
    metrics_payload = metrics.to_dict()
    validate_run_manifest_payload(manifest_payload)
    validate_metric_record_payload(metrics_payload)

    manifests_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / f"{manifest.run_id}.json"
    metrics_path = metrics_dir / f"{metrics.run_id}.json"

    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), "utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), "utf-8")
    return manifest_path, metrics_path
