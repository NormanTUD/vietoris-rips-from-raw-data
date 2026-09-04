import numpy as np
import pytest

from vrtda import PointSet
from vrtda.errors import DataError, ShapeError


def test_basic_shape():
    ps = PointSet(np.arange(6.0).reshape(3, 2))
    assert ps.n == 3
    assert ps.dim == 2
    assert len(ps) == 3


def test_rejects_1d():
    with pytest.raises(ShapeError):
        PointSet(np.arange(5.0))


def test_rejects_nan():
    x = np.ones((3, 2))
    x[0, 0] = np.nan
    with pytest.raises(DataError):
        PointSet(x)


def test_labels_len_must_match():
    with pytest.raises(ShapeError):
        PointSet(np.ones((3, 2)), labels=["a", "b"])


def test_default_labels():
    ps = PointSet(np.ones((4, 2)))
    assert ps.labels == ["0", "1", "2", "3"]


def test_select_dims():
    ps = PointSet(np.arange(12.0).reshape(3, 4))
    sub = ps.select_dims([3, 1])
    assert sub.dim == 2
    np.testing.assert_allclose(sub.data[:, 0], ps.data[:, 3])
    np.testing.assert_allclose(sub.data[:, 1], ps.data[:, 1])
    assert sub.labels == ps.labels


def test_select_dims_out_of_range():
    ps = PointSet(np.ones((3, 2)))
    with pytest.raises(AssertionError):
        ps.select_dims([5])


def test_select_rows():
    ps = PointSet(np.arange(6.0).reshape(3, 2), labels=["a", "b", "c"])
    sub = ps.select_rows([2, 0])
    assert sub.n == 2
    assert sub.labels == ["c", "a"]
    np.testing.assert_allclose(sub.data[0], ps.data[2])


def test_concat():
    a = PointSet(np.ones((2, 3)))
    b = PointSet(np.zeros((3, 3)))
    c = PointSet.concat([a, b])
    assert c.n == 5
    assert c.dim == 3
    with pytest.raises(AssertionError):
        PointSet.concat([a, PointSet(np.ones((2, 4)))])


def test_normalize_unit():
    ps = PointSet(np.array([[3.0, 4.0], [0.0, 0.0]]))
    u = ps.normalize("unit")
    np.testing.assert_allclose(np.linalg.norm(u.data[0]), 1.0)
    assert np.allclose(u.data[1], 0.0)


def test_csv_roundtrip(tmp_path):
    ps = PointSet(np.array([[1.0, 2.0], [3.0, 4.0]]))
    p = ps.to_csv(tmp_path / "x.csv")
    back = PointSet.from_csv(p)
    assert back.n == 2 and back.dim == 2
    np.testing.assert_allclose(back.data, ps.data, rtol=1e-6)


def test_from_csv_with_index(tmp_path):
    import csv
    p = tmp_path / "d.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "x", "y"])
        w.writerow(["0", "1.0", "2.0"])
        w.writerow(["1", "3.0", "4.0"])
    ps = PointSet.from_csv(p, value_cols=["x", "y"], index_cols=["idx"])
    assert ps.data.shape == (2, 2)
    assert ps.labels == ["0", "1"]


def test_from_csv_missing_file():
    with pytest.raises(DataError):
        PointSet.from_csv("/nonexistent/nope.csv")
