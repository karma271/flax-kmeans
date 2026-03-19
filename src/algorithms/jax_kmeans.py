"""JAX implementation of Lloyd's KMeans algorithm."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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


def _assign_labels(x_jax: jax.Array, centroids: jax.Array) -> jax.Array:
    """Assign each sample to the nearest centroid."""
    distances = jnp.sum((x_jax[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return jnp.argmin(distances, axis=1)


def _compute_inertia(x_jax: jax.Array, centroids: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute within-cluster sum of squared distances."""
    residuals = x_jax - centroids[labels]
    return jnp.sum(residuals * residuals)


def _update_centroids(x_jax: jax.Array, labels: jax.Array, prev_centroids: jax.Array) -> jax.Array:
    """Update centroids with mean of assigned points."""
    n_clusters = prev_centroids.shape[0]
    one_hot = jax.nn.one_hot(labels, n_clusters, dtype=x_jax.dtype)
    counts = jnp.sum(one_hot, axis=0)[:, None]
    sums = one_hot.T @ x_jax
    new_centroids = sums / jnp.maximum(counts, 1.0)
    return jnp.where(counts > 0.0, new_centroids, prev_centroids)


def _run_single_kmeans(
    x_jax: jax.Array,
    initial_centroids: jax.Array,
    max_iter: int,
    tolerance: float,
) -> tuple[jax.Array, jax.Array, jax.Array, bool, int]:
    """Run one KMeans initialization and return final state."""
    centroids = initial_centroids
    converged = False
    iterations_used = 0

    for iteration in range(max_iter):
        labels = _assign_labels(x_jax, centroids)
        updated = _update_centroids(x_jax, labels, centroids)
        shift = float(jnp.max(jnp.linalg.norm(updated - centroids, axis=1)))
        centroids = updated
        iterations_used = iteration + 1
        if shift <= tolerance:
            converged = True
            break

    final_labels = _assign_labels(x_jax, centroids)
    inertia = _compute_inertia(x_jax, centroids, final_labels)
    return centroids, final_labels, inertia, converged, iterations_used


def fit_jax_kmeans(
    x: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    random_seed: int = 0,
    init_centroids: np.ndarray | None = None,
) -> KMeansFitResult:
    """Fit KMeans using JAX and return a consistent result payload."""
    _validate_inputs(x, n_clusters, max_iter, tolerance, n_init, init_centroids)

    x_array = np.asarray(x, dtype=np.float32)
    x_jax = jnp.asarray(x_array)

    key = jax.random.PRNGKey(random_seed)
    best_result: tuple[jax.Array, jax.Array, jax.Array, bool, int] | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        if init_centroids is not None:
            initial = jnp.asarray(init_centroids, dtype=x_jax.dtype)
        else:
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey,
                x_jax.shape[0],
                shape=(n_clusters,),
                replace=False,
            )
            initial = x_jax[indices]

        result = _run_single_kmeans(
            x_jax=x_jax,
            initial_centroids=initial,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        inertia_value = float(result[2])
        if inertia_value < best_inertia:
            best_inertia = inertia_value
            best_result = result

        if init_centroids is not None:
            # Explicit centroids force a single effective initialization.
            break

    if best_result is None:
        raise RuntimeError("KMeans fitting failed to produce a result.")

    centroids, labels, inertia, converged, iterations_used = best_result
    return KMeansFitResult(
        centroids=np.asarray(centroids),
        labels=np.asarray(labels, dtype=np.int32),
        inertia=float(inertia),
        n_iter=iterations_used,
        converged=converged,
    )
