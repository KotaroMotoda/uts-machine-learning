"""Deterministic clean and noisy XOR datasets."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class NoisyXORData:
    X: FloatArray
    y: IntArray
    axis_crossing_count: int
    axis_crossing_rate: float


@dataclass(frozen=True, eq=False)
class DataSplit:
    X_train: FloatArray
    X_test: FloatArray
    y_train: IntArray
    y_test: IntArray


def clean_xor() -> tuple[FloatArray, IntArray]:
    """Return the four XOR corners and labels in {-1, +1}."""

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    y = np.array([-1, 1, 1, -1], dtype=np.int64)
    return X, y


def make_noisy_xor_clusters(
    samples_per_corner: int = 50,
    noise_std: float = 0.2,
    *,
    seed: int = 0,
) -> NoisyXORData:
    """Sample balanced Gaussian clusters around the four XOR corners."""

    if samples_per_corner < 1:
        raise ValueError("samples_per_corner must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    corners, corner_labels = clean_xor()
    rng = np.random.default_rng(seed)
    samples = []
    labels = []
    for corner, label in zip(corners, corner_labels):
        samples.append(
            rng.normal(
                loc=corner,
                scale=noise_std,
                size=(samples_per_corner, corner.shape[0]),
            )
        )
        labels.append(np.full(samples_per_corner, label, dtype=np.int64))

    X = np.vstack(samples).astype(np.float64)
    y = np.concatenate(labels)
    products = X[:, 0] * X[:, 1]
    ideal_labels = np.where(products <= 0.0, 1, -1)
    crossing_count = int(np.count_nonzero(ideal_labels != y))

    return NoisyXORData(
        X=X,
        y=y,
        axis_crossing_count=crossing_count,
        axis_crossing_rate=crossing_count / y.size,
    )


def stratified_split(
    X: NDArray[np.generic],
    y: NDArray[np.generic],
    *,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> DataSplit:
    """Split each class independently so both subsets preserve the classes."""

    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("at least two classes are required")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in classes:
        indices = np.flatnonzero(labels == label)
        if indices.size < 2:
            raise ValueError("each class needs at least two samples")
        shuffled = rng.permutation(indices)
        test_count = int(round(indices.size * test_fraction))
        test_count = min(max(test_count, 1), indices.size - 1)
        test_parts.append(shuffled[:test_count])
        train_parts.append(shuffled[test_count:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return DataSplit(
        X_train=features[train_indices].copy(),
        X_test=features[test_indices].copy(),
        y_train=labels[train_indices].copy(),
        y_test=labels[test_indices].copy(),
    )
