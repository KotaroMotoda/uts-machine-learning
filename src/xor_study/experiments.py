"""Pure orchestration for the clean and noisy XOR experiments."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from xor_study.data import clean_xor, make_noisy_xor_clusters, stratified_split
from xor_study.features import FEATURE_MAPS, transform_features
from xor_study.perceptron import empirical_error, fit_perceptron, predict_labels


DEFAULT_TRAINING_SEEDS = tuple(range(20))


@dataclass(frozen=True)
class ExperimentResult:
    feature_map: str
    seeds: Tuple[int, ...]
    convergence_rate: float
    mean_updates: float
    std_updates: float
    mean_epochs: float
    std_epochs: float
    mean_train_error: float
    std_train_error: float
    mean_test_error: Optional[float]
    std_test_error: Optional[float]
    weights: Tuple[Tuple[float, ...], ...]
    mistakes_per_epoch: Tuple[Tuple[int, ...], ...]
    updates: Tuple[int, ...]
    epochs: Tuple[int, ...]


@dataclass(frozen=True)
class NoisyExperiment:
    seed: int
    samples_per_corner: int
    noise_std: float
    test_fraction: float
    train_samples: int
    test_samples: int
    axis_crossing_count: int
    axis_crossing_rate: float
    results: Tuple[ExperimentResult, ...]


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _run_repeated(
    feature_map: str,
    X_train: NDArray[np.generic],
    y_train: NDArray[np.generic],
    *,
    seeds: Sequence[int],
    max_epochs: int,
    test_data: Optional[tuple[NDArray[np.generic], NDArray[np.generic]]] = None,
) -> ExperimentResult:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("at least one training seed is required")

    train_features = transform_features(X_train, feature_map)
    test_features = None
    test_labels = None
    if test_data is not None:
        X_test, y_test = test_data
        test_features = transform_features(X_test, feature_map)
        test_labels = np.asarray(y_test, dtype=np.int64)

    runs = [
        fit_perceptron(
            train_features,
            y_train,
            learning_rate=1.0,
            max_epochs=max_epochs,
            seed=seed,
        )
        for seed in seed_tuple
    ]
    train_errors = [run.empirical_error for run in runs]
    test_errors = None
    if test_features is not None and test_labels is not None:
        test_errors = [
            empirical_error(test_labels, predict_labels(test_features, run.weights))
            for run in runs
        ]

    mean_updates, std_updates = _mean_std([run.updates for run in runs])
    mean_epochs, std_epochs = _mean_std([run.epochs_run for run in runs])
    mean_train_error, std_train_error = _mean_std(train_errors)
    if test_errors is None:
        mean_test_error = None
        std_test_error = None
    else:
        mean_test_error, std_test_error = _mean_std(test_errors)

    return ExperimentResult(
        feature_map=feature_map,
        seeds=seed_tuple,
        convergence_rate=float(np.mean([run.converged for run in runs])),
        mean_updates=mean_updates,
        std_updates=std_updates,
        mean_epochs=mean_epochs,
        std_epochs=std_epochs,
        mean_train_error=mean_train_error,
        std_train_error=std_train_error,
        mean_test_error=mean_test_error,
        std_test_error=std_test_error,
        weights=tuple(tuple(float(value) for value in run.weights) for run in runs),
        mistakes_per_epoch=tuple(run.mistakes_per_epoch for run in runs),
        updates=tuple(run.updates for run in runs),
        epochs=tuple(run.epochs_run for run in runs),
    )


def run_clean_experiment(
    *,
    seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    max_epochs: int = 100,
) -> Tuple[ExperimentResult, ...]:
    """Run every feature map on the exact four-point XOR dataset."""

    X, y = clean_xor()
    return tuple(
        _run_repeated(name, X, y, seeds=seeds, max_epochs=max_epochs)
        for name in FEATURE_MAPS
    )


def run_noisy_experiment(
    *,
    seed: int = 13,
    samples_per_corner: int = 100,
    noise_std: float = 0.35,
    test_fraction: float = 0.25,
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    max_epochs: int = 200,
) -> NoisyExperiment:
    """Compare raw and interaction features on one reproducible noisy split."""

    data = make_noisy_xor_clusters(
        samples_per_corner=samples_per_corner,
        noise_std=noise_std,
        seed=seed,
    )
    split = stratified_split(
        data.X,
        data.y,
        test_fraction=test_fraction,
        seed=seed,
    )
    results = tuple(
        _run_repeated(
            name,
            split.X_train,
            split.y_train,
            test_data=(split.X_test, split.y_test),
            seeds=training_seeds,
            max_epochs=max_epochs,
        )
        for name in ("raw", "interaction")
    )
    return NoisyExperiment(
        seed=seed,
        samples_per_corner=samples_per_corner,
        noise_std=noise_std,
        test_fraction=test_fraction,
        train_samples=split.y_train.size,
        test_samples=split.y_test.size,
        axis_crossing_count=data.axis_crossing_count,
        axis_crossing_rate=data.axis_crossing_rate,
        results=results,
    )
