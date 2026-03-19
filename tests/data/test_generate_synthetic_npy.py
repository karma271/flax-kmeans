"""Tests for synthetic .npy generation helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.generate_synthetic_npy import generate_synthetic_blobs, run_from_args


def test_generate_synthetic_blobs_shapes_and_dtypes() -> None:
    """Generated arrays follow expected shape and dtype contracts."""
    x, y = generate_synthetic_blobs(
        n_samples=120,
        n_features=16,
        n_clusters=4,
        cluster_std=0.7,
        random_seed=17,
    )
    assert x.shape == (120, 16)
    assert y.shape == (120,)
    assert x.dtype == np.float32
    assert y.dtype == np.int32


def test_run_from_args_writes_npy_files(tmp_path: Path) -> None:
    """CLI helper writes requested output arrays."""
    x_out = tmp_path / "data" / "x.npy"
    y_out = tmp_path / "data" / "y.npy"
    x_path, y_path = run_from_args(
        [
            "--x-out",
            str(x_out),
            "--y-out",
            str(y_out),
            "--n-samples",
            "90",
            "--n-features",
            "12",
            "--n-clusters",
            "3",
            "--random-seed",
            "9",
        ]
    )

    assert x_path.exists()
    assert y_path is not None
    assert y_path.exists()

    x_loaded = np.load(x_path)
    y_loaded = np.load(y_path)
    assert x_loaded.shape == (90, 12)
    assert y_loaded.shape == (90,)
