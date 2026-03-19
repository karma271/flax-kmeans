"""sklearn KMeans wrapper with project-aligned output contract."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from src.algorithms.result import KMeansFitResult


def _validate_inputs(
    x: np.ndarray,
    n_clusters: int,
    max_iter: int,
    tolerance: float,
    n_init: int,
    init_centroids: np.ndarray | None,
) -> None:
    """Validate KMeans inputs before fitting."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features).")

    n_samples = x.shape[0]
    if n_samples == 0:
        raise ValueError("x must contain at least one sample.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > n_samples:
        raise ValueError("n_clusters must be <= number of samples.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    if n_init <= 0:
        raise ValueError("n_init must be a positive integer.")
    if init_centroids is not None and init_centroids.shape != (n_clusters, x.shape[1]):
        raise ValueError("init_centroids must have shape (n_clusters, n_features).")


def fit_sklearn_kmeans(
    x: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    random_seed: int = 0,
    init_centroids: np.ndarray | None = None,
) -> KMeansFitResult:
    """Fit KMeans via sklearn and return a consistent result payload."""
    _validate_inputs(x, n_clusters, max_iter, tolerance, n_init, init_centroids)
    x_array = np.asarray(x, dtype=np.float32)
    init: str | np.ndarray

    if init_centroids is None:
        init = "k-means++"
        effective_n_init = n_init
    else:
        init = np.asarray(init_centroids, dtype=np.float32)
        effective_n_init = 1

    estimator = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=effective_n_init,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_seed,
        algorithm="lloyd",
    )
    estimator.fit(x_array)

    return KMeansFitResult(
        centroids=np.asarray(estimator.cluster_centers_, dtype=np.float32),
        labels=np.asarray(estimator.labels_, dtype=np.int32),
        inertia=float(estimator.inertia_),
        n_iter=int(estimator.n_iter_),
        converged=bool(estimator.n_iter_ < max_iter),
    )
