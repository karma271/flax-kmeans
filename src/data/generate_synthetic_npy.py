"""Generate synthetic clustering arrays and save them as .npy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs


def generate_synthetic_blobs(
    *,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    cluster_std: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible Gaussian blob dataset for clustering."""
    x, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_seed,
    )
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for synthetic .npy generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic blobs and store arrays as .npy files."
    )
    parser.add_argument("--x-out", required=True, help="Output path for feature matrix .npy")
    parser.add_argument(
        "--y-out",
        default=None,
        help="Optional output path for labels .npy (useful for diagnostics).",
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of rows.")
    parser.add_argument("--n-features", type=int, default=32, help="Number of columns.")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of clusters.")
    parser.add_argument(
        "--cluster-std",
        type=float,
        default=0.8,
        help="Within-cluster standard deviation.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    return parser


def run_from_args(argv: list[str] | None = None) -> tuple[Path, Path | None]:
    """Generate dataset arrays and write .npy files."""
    args = build_parser().parse_args(argv)
    x, y = generate_synthetic_blobs(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
        random_seed=args.random_seed,
    )

    x_path = Path(args.x_out)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, x)

    y_path: Path | None = None
    if args.y_out is not None:
        y_path = Path(args.y_out)
        y_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(y_path, y)

    return x_path, y_path


def main() -> None:
    """Entrypoint for python -m invocation."""
    run_from_args()


if __name__ == "__main__":
    main()
