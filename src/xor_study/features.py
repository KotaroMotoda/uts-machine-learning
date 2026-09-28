"""Feature maps used to study linear separability of XOR."""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeatureMap:
    """A deterministic mapping from two inputs to a labelled feature space."""

    feature_names: Tuple[str, ...]
    transform: Callable[[FloatArray], FloatArray]


def _raw(X: FloatArray) -> FloatArray:
    return np.column_stack((np.ones(X.shape[0]), X))


def _square(X: FloatArray) -> FloatArray:
    return np.column_stack((np.ones(X.shape[0]), X, X[:, 0] ** 2, X[:, 1] ** 2))


def _interaction(X: FloatArray) -> FloatArray:
    return np.column_stack((np.ones(X.shape[0]), X, X[:, 0] * X[:, 1]))


def _quadratic(X: FloatArray) -> FloatArray:
    return np.column_stack(
        (
            np.ones(X.shape[0]),
            X,
            X[:, 0] ** 2,
            X[:, 1] ** 2,
            X[:, 0] * X[:, 1],
        )
    )


FEATURE_MAPS: Dict[str, FeatureMap] = {
    "raw": FeatureMap(("1", "x1", "x2"), _raw),
    "square": FeatureMap(("1", "x1", "x2", "x1^2", "x2^2"), _square),
    "interaction": FeatureMap(("1", "x1", "x2", "x1*x2"), _interaction),
    "quadratic": FeatureMap(
        ("1", "x1", "x2", "x1^2", "x2^2", "x1*x2"), _quadratic
    ),
}


def transform_features(X: NDArray[np.generic], name: str) -> FloatArray:
    """Apply a named feature map and include an explicit intercept column."""

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2)")

    try:
        feature_map = FEATURE_MAPS[name]
    except KeyError as error:
        available = ", ".join(FEATURE_MAPS)
        raise ValueError(f"Unknown feature map {name!r}; choose from {available}") from error

    return np.asarray(feature_map.transform(values), dtype=np.float64)
