from __future__ import annotations

import numpy as np
import pytest

from vrtda import datasets, PointSet
from vrtda.beartype_guard import beartype_module

DATA = datasets._data_root()
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="capital_berlin_multilingual data not present")


def test_list_layers() -> None:
    layers = datasets.list_layers()
    assert layers[0] == 0
    assert layers[-1] == 64
    assert len(layers) == 65


def test_load_token_cloud_shape() -> None:
    ps = datasets.load_token_cloud(layer=0)
    assert ps.n == 81
    assert ps.dim == 5120
    assert len(ps.labels) == 81
    # labels encode prompt_pos
    assert "_" in ps.labels[0]


def test_load_token_cloud_dim_selection() -> None:
    ps = datasets.load_token_cloud(layer=0).select_dims([0, 1, 2])
    assert ps.dim == 3


def test_load_layer_points() -> None:
    lp = datasets.load_layer_points(layers=[0, 16, 32])
    assert lp.n == 81 * 3
    assert lp.dim == 5120
    assert lp.labels[0].endswith("_L000")
    assert lp.labels[81].endswith("_L016")
    assert lp.labels[-1].endswith("_L032")
    # labels unique
    assert len(set(lp.labels)) == lp.n


def test_load_residual_matrix_norms() -> None:
    mat, labels = datasets.load_residual_matrix(kind="norms")
    assert mat.shape == (81, 65)
    assert len(labels) == 81
    assert np.all(mat >= 0)  # norms are non-negative


def test_load_residual_matrix_kinds() -> None:
    # norms: one value per layer (65); cosines/deltas: per consecutive transition (64)
    expected = {"norms": 65, "cosines": 64, "deltas": 64}
    for kind, ncols in expected.items():
        mat, labels = datasets.load_residual_matrix(kind=kind)
        assert mat.shape[0] == 81
        assert mat.shape[1] == ncols


def test_load_residual_matrix_bad_kind() -> None:
    from vrtda.errors import DataError
    with pytest.raises(DataError):
        datasets.load_residual_matrix(kind="bogus")


def test_token_texts() -> None:
    txt = datasets.token_texts(layer=0)
    assert len(txt) == 81
    assert all(isinstance(t, str) for t in txt)


def test_token_cloud_is_finite() -> None:
    ps = datasets.load_token_cloud(layer=32)
    assert np.all(np.isfinite(ps.data))


beartype_module(__name__)
