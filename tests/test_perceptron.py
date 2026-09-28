import numpy as np
import pytest

from xor_study.features import transform_features
from xor_study.perceptron import (
    empirical_error,
    fit_perceptron,
    predict_labels,
    predict_scores,
)


X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
y = np.array([-1, 1, 1, -1], dtype=int)


def test_interaction_features_converge_to_zero_training_error():
    Phi = transform_features(X, "interaction")
    result = fit_perceptron(Phi, y, learning_rate=1.0, max_epochs=100, seed=7)

    assert result.converged is True
    assert result.empirical_error == 0.0
    np.testing.assert_array_equal(predict_labels(Phi, result.weights), y)


def test_raw_xor_stops_without_false_convergence():
    Phi = transform_features(X, "raw")
    result = fit_perceptron(Phi, y, learning_rate=1.0, max_epochs=25, seed=7)

    assert result.converged is False
    assert result.epochs_run == 25


def test_equal_seed_reproduces_weights_and_history():
    Phi = transform_features(X, "interaction")
    first = fit_perceptron(Phi, y, seed=11)
    second = fit_perceptron(Phi, y, seed=11)

    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.mistakes_per_epoch == second.mistakes_per_epoch


def test_zero_score_is_predicted_as_positive_class():
    assert predict_labels(np.ones((1, 2)), np.zeros(2)).tolist() == [1]


def test_empirical_error_counts_wrong_labels():
    assert empirical_error(np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])) == 0.5


def test_score_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match="feature width"):
        predict_scores(np.ones((2, 3)), np.ones(4))


@pytest.mark.parametrize("bad_labels", [[0, 1, 1, -1], [-1, 1]])
def test_training_rejects_invalid_labels(bad_labels):
    with pytest.raises(ValueError, match="labels"):
        fit_perceptron(transform_features(X, "raw"), np.array(bad_labels))
