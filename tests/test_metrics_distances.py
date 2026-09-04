import numpy as np
import pytest

from vrtda import pairwise_distances, metric_names
from vrtda.errors import MetricError


def test_metric_names_present():
    names = metric_names()
    for m in ["euclidean", "squared", "cosine", "normalized_euclidean", "manhattan"]:
        assert m in names


def test_unknown_metric():
    with pytest.raises(MetricError):
        pairwise_distances(np.ones((3, 2)), metric="nope")


def test_euclidean_known():
    x = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    D = pairwise_distances(x, "euclidean")
    assert D.shape == (3, 3)
    assert D[0, 1] == pytest.approx(5.0)
    assert D[0, 2] == pytest.approx(1.0)
    assert D[1, 2] == pytest.approx(3.0 * np.sqrt(2.0))
    np.testing.assert_allclose(np.diag(D), 0.0)


def test_symmetry_and_triangle():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 7))
    D = pairwise_distances(x, "euclidean")
    np.testing.assert_allclose(D, D.T, atol=1e-9)
    # triangle inequality on a sample of triples
    for i, j, k in [(0, 1, 2), (3, 7, 9), (10, 20, 30)]:
        assert D[i, j] <= D[i, k] + D[k, j] + 1e-9


def test_squared_is_euclidean_sq():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 4))
    De = pairwise_distances(x, "euclidean")
    Ds = pairwise_distances(x, "squared")
    np.testing.assert_allclose(Ds, De ** 2, atol=1e-9)


def test_cosine_properties():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    D = pairwise_distances(x, "cosine")
    assert D[0, 1] == pytest.approx(1.0)  # orthogonal
    assert D[0, 3] == pytest.approx(0.0)  # same direction
    assert D[0, 2] == pytest.approx(1 - 1 / np.sqrt(2))


def test_normalized_euclidean_global_scale_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(15, 6))
    y = x * 3.7  # global positive scaling preserves directions
    D1 = pairwise_distances(x, "normalized_euclidean")
    D2 = pairwise_distances(y, "normalized_euclidean")
    np.testing.assert_allclose(D1, D2, atol=1e-9)


def test_normalized_euclidean_equals_euclidean_on_unit_points():
    # for already-unit-norm points the metric reduces to plain euclidean
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 4))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    D1 = pairwise_distances(x, "normalized_euclidean")
    D2 = pairwise_distances(x, "euclidean")
    np.testing.assert_allclose(D1, D2, atol=1e-9)


def test_manhattan_known():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    D = pairwise_distances(x, "manhattan")
    assert D[0, 1] == pytest.approx(2.0)


def test_invalid_input_dim():
    with pytest.raises(Exception):
        pairwise_distances(np.ones(5), "euclidean")
