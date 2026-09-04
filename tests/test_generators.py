from __future__ import annotations

import numpy as np
import pytest

from vrtda import generators as G
from vrtda.beartype_guard import beartype_module


def test_circle_shape() -> None:
    x = G.circle(20, radius=2.0)
    assert x.shape == (20, 2)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 2.0, atol=1e-9)


def test_product_torus_dim() -> None:
    assert G.product_torus(2, n=16).shape == (16, 4)
    assert G.product_torus(3, n=27).shape == (27, 6)
    assert G.product_torus(1, n=7).shape == (7, 2)


def test_product_torus_bounds() -> None:
    x = G.product_torus(2, n=50, radius=1.0, seed=0)
    assert x.min() >= -1 - 1e-9
    assert x.max() <= 1 + 1e-9


def test_donut_is_3d() -> None:
    x = G.donut(40)
    assert x.shape == (40, 3)


def test_torus3d_is_4d() -> None:
    x = G.torus3d(40)
    assert x.shape == (40, 4)


def test_sphere_surface() -> None:
    x = G.sphere(50, dim=2)
    assert x.shape == (50, 3)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)


def test_sphere_dim_param() -> None:
    x = G.sphere(30, dim=4)
    assert x.shape == (30, 5)


def test_two_blobs() -> None:
    x = G.two_blobs(n_each=20, sep=6.0, seed=0)
    assert x.shape == (40, 2)
    # two clusters along x
    assert x[:, 0].min() < 0 < x[:, 0].max()


def test_gmm() -> None:
    x = G.gmm(n_clusters=3, n_per=20, dim=4, seed=1)
    assert x.shape == (60, 4)


def test_binads() -> None:
    x = G.binads(64, dim=8, seed=2)
    assert x.shape == (64, 8)
    assert set(np.unique(x)).issubset({0.0, 1.0})


def test_grid_variants_shapes() -> None:
    assert G.circle_grid(12).shape == (12, 2)
    assert G.product_torus_grid(2, 5).shape == (25, 4)
    assert G.product_torus_grid(3, 3).shape == (27, 6)
    assert G.donut_grid(8, 5).shape == (40, 3)
    assert G.torus3d_grid(4, 3, 3).shape == (36, 4)


def test_grid_circle_radius() -> None:
    x = G.circle_grid(16, radius=1.5)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.5, atol=1e-9)


def test_donut_grid_annulus() -> None:
    x = G.donut_grid(10, 6, R=3.0, r=1.0)
    rho = np.linalg.norm(x[:, :2], axis=1)
    assert (rho >= 3.0 - 1.0 - 1e-9).all()
    assert (rho <= 3.0 + 1.0 + 1e-9).all()


def test_product_torus_grid_is_grid() -> None:
    x = G.product_torus_grid(2, 5)
    # x-coord is cos of a 5-point grid; due to cos symmetry there are <= 5 distinct values
    assert x.shape == (25, 4)
    assert len(np.unique(np.round(x[:, 0], 6))) <= 5
    # all x-values lie on the unit circle grid (radius 1)
    np.testing.assert_allclose(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1.0, atol=1e-9)


beartype_module(__name__)
