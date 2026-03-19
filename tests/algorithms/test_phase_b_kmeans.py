"""Phase B correctness tests for JAX and sklearn KMeans baselines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src.algorithms.sklearn_kmeans import fit_sklearn_kmeans


def _match_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean centroid distance after optimal bipartite matching."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)
    return float(np.mean(cost[row_idx, col_idx]))


def test_jax_kmeans_converges_on_simple_blobs() -> None:
    """JAX implementation converges and returns expected output shapes."""
    pytest.importorskip("jax")
    from src.algorithms.jax_kmeans import fit_jax_kmeans

    x, _ = make_blobs(
        n_samples=180,
        centers=[(-6.0, -6.0), (0.0, 7.5), (7.0, -2.0)],
        cluster_std=0.35,
        random_state=7,
    )
    result = fit_jax_kmeans(
        x=x,
        n_clusters=3,
        max_iter=120,
        tolerance=1e-5,
        n_init=8,
        random_seed=21,
    )

    assert result.centroids.shape == (3, 2)
    assert result.labels.shape == (180,)
    assert np.isfinite(result.inertia)
    assert result.n_iter > 0
    assert result.n_iter <= 120
    assert result.converged is True


def test_jax_and_sklearn_match_with_shared_initialization() -> None:
    """JAX and sklearn produce equivalent clustering with same init."""
    pytest.importorskip("jax")
    from src.algorithms.jax_kmeans import fit_jax_kmeans

    x, y_true = make_blobs(
        n_samples=150,
        centers=[(-8.0, -8.0), (-1.0, 5.0), (8.0, -1.0)],
        cluster_std=0.28,
        random_state=123,
    )
    x = np.asarray(x, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    # Build deterministic initial centroids (one seed point from each true cluster).
    init = np.stack([x[np.where(y_true == i)[0][0]] for i in range(3)], axis=0)

    jax_result = fit_jax_kmeans(
        x=x,
        n_clusters=3,
        max_iter=120,
        tolerance=1e-7,
        n_init=1,
        random_seed=0,
        init_centroids=init,
    )
    sklearn_result = fit_sklearn_kmeans(
        x=x,
        n_clusters=3,
        max_iter=120,
        tolerance=1e-7,
        n_init=1,
        random_seed=0,
        init_centroids=init,
    )

    assert pytest.approx(jax_result.inertia, rel=1e-5, abs=1e-5) == sklearn_result.inertia
    assert _match_center_distance(jax_result.centroids, sklearn_result.centroids) < 1e-3
    assert adjusted_rand_score(jax_result.labels, sklearn_result.labels) > 0.999
