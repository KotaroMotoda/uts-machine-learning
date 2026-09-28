import numpy as np
import pytest

from xor_study.data import clean_xor, make_noisy_xor_clusters, stratified_split


def test_clean_xor_has_four_balanced_points():
    X, y = clean_xor()

    assert X.shape == (4, 2)
    assert sorted(y.tolist()) == [-1, -1, 1, 1]


def test_noisy_clusters_are_reproducible_and_report_crossings():
    first = make_noisy_xor_clusters(samples_per_corner=25, noise_std=0.2, seed=3)
    second = make_noisy_xor_clusters(samples_per_corner=25, noise_std=0.2, seed=3)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.axis_crossing_count >= 0
    assert first.axis_crossing_rate == first.axis_crossing_count / 100


def test_zero_noise_keeps_every_point_at_its_corner():
    data = make_noisy_xor_clusters(samples_per_corner=2, noise_std=0.0, seed=7)

    assert data.axis_crossing_count == 0
    assert len(np.unique(data.X, axis=0)) == 4


def test_stratified_split_keeps_both_classes_in_both_sets():
    data = make_noisy_xor_clusters(samples_per_corner=20, noise_std=0.15, seed=5)
    split = stratified_split(data.X, data.y, test_fraction=0.25, seed=5)

    assert set(split.y_train) == {-1, 1}
    assert set(split.y_test) == {-1, 1}
    assert split.X_train.shape == (60, 2)
    assert split.X_test.shape == (20, 2)


@pytest.mark.parametrize(
    ("samples_per_corner", "noise_std"), [(0, 0.1), (5, -0.1)]
)
def test_noisy_data_rejects_invalid_parameters(samples_per_corner, noise_std):
    with pytest.raises(ValueError):
        make_noisy_xor_clusters(samples_per_corner, noise_std, seed=0)


def test_split_rejects_fraction_outside_open_unit_interval():
    X, y = clean_xor()
    with pytest.raises(ValueError, match="test_fraction"):
        stratified_split(X, y, test_fraction=1.0, seed=0)
