from __future__ import annotations

import numpy as np
import pytest

from vrtda import pairwise_distances, build_rips, build_vietoris, FilteredComplex
from vrtda.beartype_guard import beartype_module
from vrtda.errors import FiltrationError, TooLargeError


def test_vertex_count() -> None:
    pts = np.random.default_rng(0).normal(size=(10, 3))
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1e9, max_dim=0)
    assert C.count(0) == 10


def test_no_edges_below_threshold() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1.0, max_dim=1)
    assert C.count(0) == 2
    assert C.count(1) == 0


def test_edge_appears_at_distance() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 10.0, max_dim=1)
    assert C.count(1) == 1
    assert C.values[C.dims == 1][0] == pytest.approx(10.0)


def test_triangle_filtration_value_is_diameter() -> None:
    # right triangle: legs 3,4 hyp 5 -> diameter 5
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1e9, max_dim=2)
    tri = [s for (v, d, s) in zip(C.values, C.dims, C.simplexes) if d == 2]
    assert len(tri) == 1
    idx = C.index_of(tri[0])
    assert C.values[idx] == pytest.approx(5.0)


def test_faces_before_cofaces() -> None:
    pts = np.random.default_rng(1).normal(size=(12, 3))
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1.0, max_dim=3)
    for j, s in enumerate(C.simplexes):
        for face in C.boundary_faces(j):
            assert face < j, f"face {face} not before coface {j}"
            assert C.values[face] <= C.values[j] + 1e-12


def test_full_simplex_contractible() -> None:
    # 3 points, large eps -> a 2-simplex (contractible)
    pts = np.random.default_rng(2).normal(size=(3, 3))
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1e9, max_dim=2)
    from vrtda import betti_at
    assert betti_at(C, 1e9) == [1, 0, 0]


def test_too_large_raises() -> None:
    pts = np.random.default_rng(3).normal(size=(40, 2))
    D = pairwise_distances(pts)
    with pytest.raises(TooLargeError):
        build_rips(pts, D, 1e9, max_dim=2, max_simplices=100)


def test_max_dim_too_high() -> None:
    pts = np.random.default_rng(4).normal(size=(5, 3))
    D = pairwise_distances(pts)
    with pytest.raises(FiltrationError):
        build_rips(pts, D, 1.0, max_dim=4)


def test_vietoris_subset_of_rips_scale() -> None:
    # vietoris at radius r uses edges d<=2r; compare structure sanity
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(15, 2))
    D = pairwise_distances(pts)
    r = 0.5
    Cv = build_vietoris(pts, D, r, max_dim=2)
    assert Cv.count(0) == 15
    # every vietoris edge must satisfy d<=2r
    for s in Cv.simplexes:
        if len(s) == 2:
            assert D[s[0], s[1]] <= 2 * r + 1e-9
    # every vietoris triangle has MEB <= r
    from vrtda.geometry import min_enclosing_ball_radius
    for s in Cv.simplexes:
        if len(s) == 3:
            assert min_enclosing_ball_radius(pts[list(s)]) <= r + 1e-9


def test_vietoris_is_a_complex() -> None:
    rng = np.random.default_rng(6)
    pts = rng.normal(size=(15, 2))
    D = pairwise_distances(pts)
    Cv = build_vietoris(pts, D, 0.6, max_dim=2)
    for j, s in enumerate(Cv.simplexes):
        for face in Cv.boundary_faces(j):
            assert face < j


def test_summary_and_repr() -> None:
    pts = np.random.default_rng(7).normal(size=(6, 2))
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1.0, max_dim=2)
    s = C.summary()
    assert s["kind"] == "rips"
    assert s["n_simplices"] == C.n_simplices
    assert "rips" in repr(C)


beartype_module(__name__)
