"""Shared experiment contracts for reproducible benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

Device = Literal["cpu", "gpu", "tpu"]
Implementation = Literal[
    "jax_kmeans",
    "jax_flash_kmeans",
    "sklearn_kmeans",
    "flashkmeans_wrapper",
]


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration used by benchmark runners for one implementation."""

    run_id: str
    implementation: Implementation
    dataset_id: str
    random_seed: int
    n_clusters: int
    max_iter: int
    tolerance: float
    n_init: int
    batch_size: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for storage and logging."""
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    """Runtime metadata captured for every execution."""

    run_id: str
    timestamp_utc: str
    implementation: Implementation
    device: Device
    dataset_id: str
    n_samples: int
    n_features: int
    n_clusters: int
    random_seed: int
    max_iter: int
    tolerance: float
    n_init: int
    converged: bool
    iterations_used: int
    fit_time_ms: float
    predict_time_ms: float
    peak_memory_mb: float | None = None
    software_versions: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest for JSON output."""
        return asdict(self)

    @staticmethod
    def utc_now() -> str:
        """Return ISO-8601 UTC timestamp for manifests."""
        return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class MetricRecord:
    """Normalized clustering quality metrics for cross-implementation comparison."""

    run_id: str
    implementation: Implementation
    dataset_id: str
    inertia: float
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics for tabular aggregation."""
        return asdict(self)
