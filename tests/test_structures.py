"""Exact-topology validation for the known structures the tools rebuild.

Covers the ground-truth cell complexes and their Betti numbers:
  * make_simplicial_sphere(k)  -> S^k,            beta = [1, 0, ..., 0, 1]
  * make_bouquet_complex(n, k) -> wedge of n S^k, beta = [1, 0, ..., n@k]
  * make_torus_grid_complex(k) -> T^k,            beta = [C(k,0), ..., C(k,k)]
plus Vietoris-Rips recovery on the actual point clouds for the clean low-dim
cases (figure-of-n, 1-/2-torus), generator shape/determinism checks, and smoke
tests on the tools' build()/verify() entry points.
"""
from __future__ import annotations

import argparse
import importlib.util
import types
from math import comb
from pathlib import Path

import numpy as np
import pytest

from vrtda import pairwise_distances, build_rips, persistent_homology, betti_at
from vrtda import generators as G
from vrtda.complexes import (
    FilteredComplex,
    make_torus_grid_complex,
    make_simplicial_sphere,
    make_bouquet_complex,
)
from vrtda.beartype_guard import beartype_module

ROOT = Path(__file__).resolve().parents[1]


def _nn(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def _rip_betti(X: np.ndarray, frac: float = 1.6, max_dim: int = 3) -> list[int]:
    D = pairwise_distances(X)
    eps = frac * _nn(D)
    C = build_rips(X, D, eps, max_dim=max_dim)
    return persistent_homology(C).betti_at(eps)


def _euler(C: FilteredComplex) -> int:
    return sum((-1) ** d * C.count(d) for d in range(C.max_dim() + 1))


def _load_tool(name: str) -> types.ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# S^k (exact simplicial sphere)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_simplicial_sphere_betti(k: int) -> None:
    C = make_simplicial_sphere(k)
    b = betti_at(C, float(C.values.max()))
    expect = [0] * (k + 1)
    expect[0] = 1
    expect[k] = 1
    assert b == expect
    assert _euler(C) == 1 + (-1) ** k  # Euler characteristic of S^k


# --------------------------------------------------------------------------- #
# Bouquet / wedge of n k-spheres
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_bouquet_betti(n: int, k: int) -> None:
    C = make_bouquet_complex(n, k)
    b = betti_at(C, float(C.values.max()))
    expect = [0] * (k + 1)
    expect[0] = 1
    expect[k] = n
    assert b == expect


def test_bouquet_figure8() -> None:
    # n=2, k=1 is the classic figure-8: one component, two independent loops.
    C = make_bouquet_complex(2, 1)
    assert betti_at(C, float(C.values.max())) == [1, 2]


# --------------------------------------------------------------------------- #
# T^k (product torus), exact cell complex
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 3, 4])
def test_torus_exact_betti(k: int) -> None:
    C = make_torus_grid_complex(k, (3,) * k)
    b = betti_at(C, float(C.values.max()))
    assert b == [comb(k, d) for d in range(k + 1)]
    assert _euler(C) == 0  # Euler characteristic of T^k is 0


def test_torus_k5_constructs_and_euler_zero() -> None:
    # k=5 exact homology is slow (~40s); verify the complex builds with the right
    # top dimension and Euler characteristic instead of computing the full Betti.
    C = make_torus_grid_complex(5, (3,) * 5)
    assert C.max_dim() == 5
    assert _euler(C) == 0


# --------------------------------------------------------------------------- #
# Vietoris-Rips recovery on the actual clouds (clean low-dimensional cases)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 4])
def test_bouquet_circles_rips(n: int) -> None:
    X = G.bouquet_circles(n, n_per=16, radius=1.0)
    b = _rip_betti(X, frac=1.6, max_dim=2)
    assert (list(b) + [0])[:2] == [1, n]


@pytest.mark.parametrize("k", [1, 2])
def test_product_torus_rips_lowdim(k: int) -> None:
    X = G.product_torus_grid(k, 8)
    b = _rip_betti(X, frac=1.6, max_dim=k + 1)
    expect = [comb(k, d) for d in range(k + 1)]
    assert (list(b) + [0] * k)[: len(expect)] == expect


# --------------------------------------------------------------------------- #
# Generator shape / determinism
# --------------------------------------------------------------------------- #
def test_bouquet_circles_shape_and_wedge_point() -> None:
    for n in (1, 2, 4):
        X = G.bouquet_circles(n, n_per=16)
        assert X.shape == (n * 16, 2 * n)
        # every circle passes through the origin: exactly n coincident all-zero rows
        origin_rows = int((X == 0).all(axis=1).sum())
        assert origin_rows == n


def test_bouquet_circles_deterministic() -> None:
    assert np.array_equal(G.bouquet_circles(3, n_per=12), G.bouquet_circles(3, n_per=12))


def test_product_torus_grid_shape() -> None:
    for k in (1, 2, 3):
        X = G.product_torus_grid(k, 6)
        assert X.shape == (6**k, 2 * k)


def test_sphere_shape_and_radius() -> None:
    X = G.sphere(50, dim=3, seed=0)
    assert X.shape == (50, 4)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


# --------------------------------------------------------------------------- #
# Tool smoke tests (build + verify entry points)
# --------------------------------------------------------------------------- #
def _ns(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        kind="product", k=2, n=24, nper=8, radius=1.0, minor=0.35, grid=True, noise=0.0, seed=0
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_tool_make_torus_build_shapes() -> None:
    t = _load_tool("make_torus")
    assert t.build(_ns()).shape == (64, 4)                # T^2 grid 8x8 in R^4
    assert t.build(_ns(kind="circle", k=1)).shape == (24, 2)   # 1-torus
    assert t.build(_ns(kind="donut")).shape == (24 * 8, 3)     # bagel grid
    assert t.build(_ns(kind="sphere", k=3, grid=False)).shape == (24, 4)


def test_tool_make_torus_verify() -> None:
    t = _load_tool("make_torus")
    t.verify("product", 3, t.build(_ns(kind="product", k=3, nper=5)))  # asserts [1,3,3,1]
    t.verify("circle", 1, t.build(_ns(kind="circle", k=1)))            # asserts [1,1]


def test_tool_make_sphere_build_and_verify() -> None:
    s = _load_tool("make_sphere")
    ns = argparse.Namespace(k=2, n=128, radius=1.0, noise=0.0, seed=0)
    assert s.build(ns).shape == (128, 3)
    s.verify(2, s.build(ns))  # asserts exact S^2 = [1,0,1]


def test_tool_make_wedge_build_and_verify() -> None:
    w = _load_tool("make_wedge")
    ns = argparse.Namespace(n=3, k=1, nper=16, radius=1.0)
    assert w.build(ns).shape == (48, 6)
    w.verify(3, 1, w.build(ns))  # asserts exact [1,3] and rips [1,3]
    w.verify(2, 2, None)         # asserts exact bouquet of 2 S^2 = [1,0,2]


beartype_module(__name__)
