"""A small, deterministic implementation of the binary Perceptron."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PerceptronResult:
    """Training result and the evidence needed to discuss convergence."""

    weights: FloatArray
    converged: bool
    epochs_run: int
    updates: int
    mistakes_per_epoch: Tuple[int, ...]
    empirical_error: float


def _as_feature_matrix(Phi: NDArray[np.generic]) -> FloatArray:
    values = np.asarray(Phi, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Phi must be a non-empty two-dimensional feature matrix")
    return values


def predict_scores(Phi: NDArray[np.generic], weights: NDArray[np.generic]) -> FloatArray:
    """Return the linear score ``Phi @ weights`` for every sample."""

    features = _as_feature_matrix(Phi)
    coefficients = np.asarray(weights, dtype=np.float64)
    if coefficients.ndim != 1 or features.shape[1] != coefficients.shape[0]:
        raise ValueError("weights must be one-dimensional and match the feature width")
    return features @ coefficients


def predict_labels(Phi: NDArray[np.generic], weights: NDArray[np.generic]) -> IntArray:
    """Predict -1 or +1; a score of exactly zero is assigned to +1."""

    scores = predict_scores(Phi, weights)
    return np.where(scores >= 0.0, 1, -1).astype(np.int64)


def empirical_error(y_true: NDArray[np.generic], y_pred: NDArray[np.generic]) -> float:
    """Return the fraction of labels predicted incorrectly."""

    expected = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if expected.ndim != 1 or predicted.ndim != 1 or expected.shape != predicted.shape:
        raise ValueError("true and predicted labels must be equally sized one-dimensional arrays")
    if expected.size == 0:
        raise ValueError("labels must not be empty")
    return float(np.mean(expected != predicted))


def fit_perceptron(
    Phi: NDArray[np.generic],
    y: NDArray[np.generic],
    *,
    learning_rate: float = 1.0,
    max_epochs: int = 100,
    seed: int = 0,
) -> PerceptronResult:
    """Fit a binary Perceptron using a reproducible shuffle each epoch."""

    features = _as_feature_matrix(Phi)
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one-dimensional and match the sample count")
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("labels must contain only -1 and +1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")

    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    mistakes_history = []
    updates = 0
    converged = False

    for _ in range(max_epochs):
        mistakes = 0
        for index in rng.permutation(features.shape[0]):
            score = float(features[index] @ weights)
            prediction = 1 if score >= 0.0 else -1
            if prediction != labels[index]:
                weights += learning_rate * labels[index] * features[index]
                updates += 1
                mistakes += 1
        mistakes_history.append(mistakes)
        if mistakes == 0:
            converged = True
            break

    predictions = predict_labels(features, weights)
    return PerceptronResult(
        weights=weights.copy(),
        converged=converged,
        epochs_run=len(mistakes_history),
        updates=updates,
        mistakes_per_epoch=tuple(mistakes_history),
        empirical_error=empirical_error(labels, predictions),
    )
