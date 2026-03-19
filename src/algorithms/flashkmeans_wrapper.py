"""Wrapper around external flash-kmeans package with project result contract."""

from __future__ import annotations

import numpy as np

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


def _compute_inertia(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute within-cluster sum of squared distances."""
    residuals = x - centroids[labels]
    return float(np.sum(residuals * residuals))


def fit_flashkmeans_wrapper(
    x: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    random_seed: int = 0,
    init_centroids: np.ndarray | None = None,
    use_triton: bool = True,
    device: str | None = None,
    dtype: str = "float32",
) -> KMeansFitResult:
    """Fit KMeans via flash-kmeans package and return a consistent result payload."""
    _validate_inputs(x, n_clusters, max_iter, tolerance, n_init, init_centroids)

    try:
        import torch
        from flash_kmeans import batch_kmeans_Euclid  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "flash-kmeans and torch are required for fit_flashkmeans_wrapper."
        ) from exc

    x_array = np.asarray(x, dtype=np.float32)
    x_tensor = torch.as_tensor(x_array)

    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be one of: 'float16', 'float32'.")

    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    effective_n_init = 1 if init_centroids is not None else n_init
    best_centroids: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_n_iter = max_iter
    best_inertia = np.inf

    x_tensor = x_tensor.to(device=target_device, dtype=torch_dtype)
    init_tensor = None
    if init_centroids is not None:
        init_tensor = torch.as_tensor(
            np.asarray(init_centroids, dtype=np.float32),
            dtype=torch_dtype,
            device=target_device,
        ).unsqueeze(0)

    # Behavior is controlled by flash-kmeans package installation/runtime.
    _ = use_triton

    for run_idx in range(effective_n_init):
        torch.manual_seed(random_seed + run_idx)
        cluster_ids, centers, n_iter_used = batch_kmeans_Euclid(
            x_tensor.unsqueeze(0),
            n_clusters=n_clusters,
            max_iters=max_iter,
            tol=tolerance,
            init_centroids=init_tensor,
            verbose=False,
        )

        labels_np = cluster_ids.squeeze(0).detach().cpu().numpy().astype(np.int32, copy=False)
        centroids_np = centers.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        inertia = _compute_inertia(x_array, labels_np, centroids_np)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = np.asarray(labels_np, dtype=np.int32)
            best_centroids = np.asarray(centroids_np, dtype=np.float32)
            best_n_iter = int(n_iter_used)

    if best_centroids is None or best_labels is None:
        raise RuntimeError("flash-kmeans fitting failed to produce a result.")

    return KMeansFitResult(
        centroids=best_centroids,
        labels=best_labels,
        inertia=best_inertia,
        n_iter=best_n_iter,
        converged=bool(best_n_iter < max_iter),
    )
