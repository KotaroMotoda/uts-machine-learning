import numpy as np
import pytest

from xor_study.features import FEATURE_MAPS, transform_features


XOR_X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)


def test_all_required_feature_maps_are_available():
    assert tuple(FEATURE_MAPS) == ("raw", "square", "interaction", "quadratic")


def test_interaction_map_adds_x1_x2():
    transformed = transform_features(XOR_X, "interaction")
    np.testing.assert_array_equal(
        transformed[:, -1], np.array([1, -1, -1, 1], dtype=float)
    )


def test_square_terms_are_constant_on_clean_xor():
    transformed = transform_features(XOR_X, "square")
    np.testing.assert_array_equal(transformed[:, -2:], np.ones((4, 2)))


@pytest.mark.parametrize(
    ("name", "expected_width"),
    [("raw", 3), ("square", 5), ("interaction", 4), ("quadratic", 6)],
)
def test_feature_maps_include_intercept_and_expected_width(name, expected_width):
    transformed = transform_features(XOR_X, name)

    assert transformed.shape == (4, expected_width)
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(transformed[:, 0], np.ones(4))


def test_unknown_feature_map_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature map"):
        transform_features(XOR_X, "missing")


@pytest.mark.parametrize("bad_shape", [np.ones(2), np.ones((3, 1)), np.ones((2, 2, 1))])
def test_inputs_must_have_two_feature_columns(bad_shape):
    with pytest.raises(ValueError, match=r"shape \(n_samples, 2\)"):
        transform_features(bad_shape, "raw")
