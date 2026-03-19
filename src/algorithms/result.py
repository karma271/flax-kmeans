"""Shared result model returned by KMeans implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KMeansFitResult:
    """Result payload returned by a KMeans fit call."""

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
