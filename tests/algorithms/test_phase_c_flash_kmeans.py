"""Phase C tests for flash-style JAX and flash-kmeans wrapper paths."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src.algorithms.jax_kmeans import fit_jax_kmeans


def _match_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean centroid distance after optimal bipartite matching."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)
    return float(np.mean(cost[row_idx, col_idx]))


def test_jax_flash_kmeans_converges_on_simple_blobs() -> None:
    """JAX flash-style implementation converges and returns expected outputs."""
    pytest.importorskip("jax")
    from src.algorithms.jax_flash_kmeans import fit_jax_flash_kmeans

    x, _ = make_blobs(
        n_samples=210,
        centers=[(-8.0, -8.0), (0.0, 7.0), (8.0, -2.0)],
        cluster_std=0.4,
        random_state=11,
    )
    result = fit_jax_flash_kmeans(
        x=x,
        n_clusters=3,
        max_iter=150,
        tolerance=1e-5,
        n_init=6,
        random_seed=9,
        data_chunk_size=64,
        centroid_chunk_size=2,
    )

    assert result.centroids.shape == (3, 2)
    assert result.labels.shape == (210,)
    assert np.isfinite(result.inertia)
    assert result.n_iter > 0
    assert result.n_iter <= 150
    assert result.converged is True


def test_jax_flash_matches_jax_lloyd_with_shared_initialization() -> None:
    """Flash-style and vanilla JAX converge to near-equivalent solutions."""
    pytest.importorskip("jax")
    from src.algorithms.jax_flash_kmeans import fit_jax_flash_kmeans

    x, y_true = make_blobs(
        n_samples=180,
        centers=[(-7.0, -6.5), (-1.0, 6.0), (7.0, -1.5)],
        cluster_std=0.3,
        random_state=1234,
    )
    x = np.asarray(x, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)
    init = np.stack([x[np.where(y_true == i)[0][0]] for i in range(3)], axis=0)

    flash_result = fit_jax_flash_kmeans(
        x=x,
        n_clusters=3,
        max_iter=120,
        tolerance=1e-7,
        n_init=1,
        random_seed=0,
        init_centroids=init,
        data_chunk_size=48,
        centroid_chunk_size=2,
    )
    vanilla_result = fit_jax_kmeans(
        x=x,
        n_clusters=3,
        max_iter=120,
        tolerance=1e-7,
        n_init=1,
        random_seed=0,
        init_centroids=init,
    )

    assert pytest.approx(flash_result.inertia, rel=1e-5, abs=1e-5) == vanilla_result.inertia
    assert _match_center_distance(flash_result.centroids, vanilla_result.centroids) < 1e-3
    assert adjusted_rand_score(flash_result.labels, vanilla_result.labels) > 0.999


def test_flashkmeans_wrapper_runs_when_dependency_available() -> None:
    """flash-kmeans wrapper returns contract output when package is available."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("flash_kmeans")
    from src.algorithms.flashkmeans_wrapper import fit_flashkmeans_wrapper

    x, _ = make_blobs(
        n_samples=90,
        centers=[(-5.0, -5.0), (0.0, 4.0), (5.0, -1.0)],
        cluster_std=0.35,
        random_state=5,
    )
    x = np.asarray(x, dtype=np.float32)

    # CPU execution keeps this test runnable in non-CUDA environments.
    result = fit_flashkmeans_wrapper(
        x=x,
        n_clusters=3,
        max_iter=50,
        tolerance=1e-5,
        n_init=2,
        random_seed=42,
        use_triton=False,
        device="cpu",
        dtype="float32",
    )

    assert result.centroids.shape == (3, 2)
    assert result.labels.shape == (90,)
    assert np.isfinite(result.inertia)
    assert result.n_iter > 0
    assert result.n_iter <= 50
    assert isinstance(result.converged, bool)
    assert result.labels.dtype == np.int32
    assert result.centroids.dtype == np.float32
    assert torch.is_tensor(torch.as_tensor(result.centroids))
