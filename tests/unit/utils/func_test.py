import numpy as np
import pytest

from utils.func import sigmoid


def test_sigmoid_of_zero_is_one_half():
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_of_known_values():
    cases = {
        0.0: 0.5,
        1.0: 0.7310585786300049,
        -1.0: 0.2689414213699951,
        2.0: 0.8807970779778823,
        -2.0: 0.11920292202211755,
    }
    for z, expected in cases.items():
        assert sigmoid(z) == pytest.approx(expected)


def test_sigmoid_accepts_and_returns_numpy_array():
    z = np.array([0.0, 1.0, -1.0])
    result = sigmoid(z)
    assert isinstance(result, np.ndarray)
    assert result.shape == z.shape


def test_sigmoid_preserves_multidimensional_shape():
    z = np.random.randn(4, 3, 2)
    result = sigmoid(z)
    assert result.shape == z.shape


def test_sigmoid_is_applied_elementwise():
    z = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    result = sigmoid(z)
    expected = np.array([sigmoid(zi) for zi in z])
    np.testing.assert_allclose(result, expected)


def test_sigmoid_output_is_always_between_zero_and_one():
    z = np.linspace(-50, 50, 1000)
    result = sigmoid(z)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_sigmoid_output_is_strictly_between_zero_and_one_for_moderate_inputs():
    z = np.linspace(-20, 20, 1000)
    result = sigmoid(z)
    assert np.all(result > 0.0)
    assert np.all(result < 1.0)


def test_sigmoid_is_monotonically_increasing():
    z = np.linspace(-10, 10, 500)
    result = sigmoid(z)

    assert np.all(np.diff(result) >= 0)


def test_sigmoid_symmetry_property():

    z = np.array([0.5, 1.5, -3.0, 10.0, -0.001])
    np.testing.assert_allclose(sigmoid(-z), 1 - sigmoid(z), rtol=1e-10)


def test_sigmoid_of_negative_and_positive_are_complementary_around_half():
    z = 3.7
    assert sigmoid(z) + sigmoid(-z) == pytest.approx(1.0)


def test_sigmoid_handles_large_positive_values_without_overflow_error():
    z = np.array([100.0, 1000.0, 1e10])
    with np.errstate(over="raise"):  # would raise if this direction overflowed
        result = sigmoid(z)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)
    assert not np.any(np.isnan(result))


def test_sigmoid_handles_large_negative_values():
    z = np.array([-100.0, -1000.0, -1e10])
    with np.errstate(over="ignore"):
        result = sigmoid(z)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)
    assert not np.any(np.isnan(result))


def test_sigmoid_large_negative_values_do_not_produce_nan():
    z = np.array([-500.0, -5000.0])
    with pytest.warns(RuntimeWarning, match="overflow"):
        result = sigmoid(z)
    assert not np.any(np.isnan(result))


def test_sigmoid_of_empty_array_returns_empty_array():
    z = np.array([])
    result = sigmoid(z)
    assert result.shape == (0,)


def test_sigmoid_accepts_python_scalar_not_just_numpy_array():
    result = sigmoid(0.0)
    assert result == pytest.approx(0.5)


def test_sigmoid_rejects_plain_python_list():
    with pytest.raises(TypeError):
        sigmoid([0.0, 1.0])


def test_sigmoid_preserves_float_dtype():
    z = np.array([0.0, 1.0], dtype=np.float32)
    result = sigmoid(z)
    assert result.dtype == np.float32


def test_sigmoid_promotes_integer_input_to_float():
    z = np.array([0, 1, -1], dtype=np.int64)
    result = sigmoid(z)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, [0.5, 0.7310585786300049, 0.2689414213699951])
