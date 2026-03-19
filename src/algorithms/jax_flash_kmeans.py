"""JAX flash-style KMeans with chunked assignment to reduce peak memory."""

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
    data_chunk_size: int,
    centroid_chunk_size: int,
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
    if data_chunk_size <= 0:
        raise ValueError("data_chunk_size must be a positive integer.")
    if centroid_chunk_size <= 0:
        raise ValueError("centroid_chunk_size must be a positive integer.")
    if init_centroids is not None and init_centroids.shape != (n_clusters, x.shape[1]):
        raise ValueError("init_centroids must have shape (n_clusters, n_features).")


def _assign_labels_chunked(
    x_jax: jax.Array,
    centroids: jax.Array,
    data_chunk_size: int,
    centroid_chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Assign labels in chunks to avoid allocating a full (N, K) distance matrix."""
    n_samples = x_jax.shape[0]
    n_clusters = centroids.shape[0]
    best_dist = jnp.full((n_samples,), jnp.inf, dtype=x_jax.dtype)
    best_labels = jnp.zeros((n_samples,), dtype=jnp.int32)

    for data_start in range(0, n_samples, data_chunk_size):
        data_end = min(data_start + data_chunk_size, n_samples)
        x_chunk = x_jax[data_start:data_end]

        chunk_best_dist = jnp.full((data_end - data_start,), jnp.inf, dtype=x_jax.dtype)
        chunk_best_labels = jnp.zeros((data_end - data_start,), dtype=jnp.int32)

        for centroid_start in range(0, n_clusters, centroid_chunk_size):
            centroid_end = min(centroid_start + centroid_chunk_size, n_clusters)
            centroid_chunk = centroids[centroid_start:centroid_end]
            distances = jnp.sum(
                (x_chunk[:, None, :] - centroid_chunk[None, :, :]) ** 2,
                axis=2,
            )
            local_best_dist = jnp.min(distances, axis=1)
            local_best_labels = (
                jnp.argmin(distances, axis=1).astype(jnp.int32) + centroid_start
            )
            better = local_best_dist < chunk_best_dist
            chunk_best_dist = jnp.where(better, local_best_dist, chunk_best_dist)
            chunk_best_labels = jnp.where(better, local_best_labels, chunk_best_labels)

        best_dist = best_dist.at[data_start:data_end].set(chunk_best_dist)
        best_labels = best_labels.at[data_start:data_end].set(chunk_best_labels)

    return best_labels, best_dist


def _update_centroids(
    x_jax: jax.Array,
    labels: jax.Array,
    prev_centroids: jax.Array,
) -> jax.Array:
    """Update centroids using bincount-based aggregation."""
    n_clusters = prev_centroids.shape[0]
    counts = jnp.bincount(labels, length=n_clusters).astype(x_jax.dtype)
    weighted_sums = jax.vmap(
        lambda col: jnp.bincount(labels, weights=col, length=n_clusters),
        in_axes=1,
        out_axes=1,
    )(x_jax)
    new_centroids = weighted_sums / jnp.maximum(counts[:, None], 1.0)
    return jnp.where(counts[:, None] > 0.0, new_centroids, prev_centroids)


def _run_single_kmeans(
    x_jax: jax.Array,
    initial_centroids: jax.Array,
    max_iter: int,
    tolerance: float,
    data_chunk_size: int,
    centroid_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, bool, int]:
    """Run one initialization and return final state."""
    centroids = initial_centroids
    converged = False
    iterations_used = 0

    for iteration in range(max_iter):
        labels, _ = _assign_labels_chunked(
            x_jax=x_jax,
            centroids=centroids,
            data_chunk_size=data_chunk_size,
            centroid_chunk_size=centroid_chunk_size,
        )
        updated = _update_centroids(x_jax, labels, centroids)
        shift = float(jnp.max(jnp.linalg.norm(updated - centroids, axis=1)))
        centroids = updated
        iterations_used = iteration + 1
        if shift <= tolerance:
            converged = True
            break

    final_labels, final_dist = _assign_labels_chunked(
        x_jax=x_jax,
        centroids=centroids,
        data_chunk_size=data_chunk_size,
        centroid_chunk_size=centroid_chunk_size,
    )
    inertia = jnp.sum(final_dist)
    return centroids, final_labels, inertia, converged, iterations_used


def fit_jax_flash_kmeans(
    x: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    random_seed: int = 0,
    init_centroids: np.ndarray | None = None,
    data_chunk_size: int = 4096,
    centroid_chunk_size: int = 256,
) -> KMeansFitResult:
    """Fit flash-style KMeans in JAX and return a consistent result payload."""
    _validate_inputs(
        x=x,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tolerance=tolerance,
        n_init=n_init,
        init_centroids=init_centroids,
        data_chunk_size=data_chunk_size,
        centroid_chunk_size=centroid_chunk_size,
    )

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
            data_chunk_size=data_chunk_size,
            centroid_chunk_size=centroid_chunk_size,
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
        centroids=np.asarray(centroids, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        inertia=float(inertia),
        n_iter=iterations_used,
        converged=converged,
    )
