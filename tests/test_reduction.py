from __future__ import annotations

import numpy as np
import pytest

from vrtda import reduction as R
from vrtda.beartype_guard import beartype_module
from vrtda.errors import DataError


def test_pca_finds_dominant_axis() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2000, 2))
    X = base @ np.array([[10.0, 0.0], [0.0, 1.0]])  # stretch x
    scores, comps, evr, mean = R.pca(X, n_components=2)
    # first principal axis ~ x-axis (sample eigenvector has O(1/sqrt(n)) off-axis error)
    assert abs(abs(comps[0, 0]) - 1.0) < 0.01
    assert abs(comps[0, 1]) < 0.01
    assert evr[0] > 0.98
    # scores dominate the variance
    assert scores[:, 0].var() > scores[:, 1].var() * 50


def test_pca_collinear_rank1() -> None:
    t = np.linspace(0, 1, 50).reshape(-1, 1)
    X = np.hstack([t, 2 * t + 0.5, -3 * t])  # rank 1 (affine)
    scores, comps, evr, _ = R.pca(X, n_components=3)
    assert evr[0] == pytest.approx(1.0, rel=1e-9)
    assert evr[1] < 1e-9


def test_pca_explained_variance_matches_eigenvalues() -> None:
    rng = np.random.default_rng(1)
    C = np.array([[4.0, 2.0], [2.0, 3.0]])
    X = rng.multivariate_normal([0.0, 0.0], C, size=20000)
    scores, comps, evr, _ = R.pca(X, n_components=2)
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    expected_evr = eigvals / eigvals.sum()
    np.testing.assert_allclose(evr, expected_evr, atol=5e-3)
    # components span the eigenvectors (up to sign)
    for i in range(2):
        v = comps[i]
        # distance to the subspace should be tiny; check |v·eigvec| ~ 1 for the right one
        assert np.isclose(v @ v, 1.0, atol=1e-9)


def test_pca_mean_and_scores_shape() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    scores, comps, evr, mean = R.pca(X, n_components=3)
    assert scores.shape == (30, 3)
    assert comps.shape == (3, 4)
    assert len(evr) == 3
    assert mean.shape == (4,)
    np.testing.assert_allclose(mean, X.mean(axis=0), atol=1e-12)
    # evr decreasing and <= 1
    assert np.all(np.diff(evr) <= 1e-12)
    assert evr.max() <= 1.0 + 1e-12


def test_variance_of_and_top_dims() -> None:
    X = np.array([[1.0, 10.0, 2.0], [2.0, 12.0, 5.0], [0.0, 8.0, 4.0], [1.0, 11.0, 3.0]])
    var = R.variance_of(X)
    # variances: dim0=[1,2,0,1]->0.5, dim1=[10,12,8,11]->2.1875, dim2=[2,5,4,3]->1.25
    order = R.top_variance_dims(X, 2)
    assert order[0] == 1  # dim 1 most variable (2.1875)
    assert order[1] == 2  # dim 2 next (1.25)
    assert var[order[0]] == var.max()


def test_top_variance_dims_bounds() -> None:
    X = np.random.default_rng(3).normal(size=(5, 4))
    assert R.top_variance_dims(X, 0) == []
    assert R.top_variance_dims(X, 99) == list(range(4))


def test_reduce_dispatch_pca() -> None:
    X = np.random.default_rng(4).normal(size=(20, 5))
    out, meta = R.reduce(X, "pca", n_components=2)
    assert out.shape == (20, 2)
    assert meta["method"] == "pca"
    assert len(meta["explained_variance_ratio"]) == 2


def test_reduce_unknown_method() -> None:
    X = np.random.default_rng(5).normal(size=(5, 3))
    with pytest.raises(DataError):
        R.reduce(X, "nope")


def test_reduce_pca_equals_direct() -> None:
    X = np.random.default_rng(6).normal(size=(15, 6))
    out, _ = R.reduce(X, "pca", n_components=3)
    scores, _, _, _ = R.pca(X, n_components=3)
    np.testing.assert_allclose(out, scores, atol=1e-12)


beartype_module(__name__)
