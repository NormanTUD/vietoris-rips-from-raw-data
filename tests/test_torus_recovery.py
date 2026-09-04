import numpy as np
import pytest

from vrtda import (
    pairwise_distances,
    build_rips,
    persistent_homology,
    betti_at,
    cohomology_at,
)
from vrtda import generators as G
from vrtda.complexes import make_torus_grid_complex


def _nearest_neighbor_mean(D):
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def _reaches_beta1(pts, max_dim, target_beta1):
    """Does there exist an epsilon where the complex is connected (beta_0=1)
    and beta_1 == target_beta1?"""
    D = pairwise_distances(pts)
    nn = _nearest_neighbor_mean(D)
    C = build_rips(pts, D, 2.5 * nn, max_dim=max_dim)
    bc = persistent_homology(C)
    epsilons = np.linspace(0.9 * nn, 2.0 * nn, 40)
    for e in epsilons:
        b = bc.betti_at(e)
        if b[0] == 1 and len(b) > 1 and b[1] == target_beta1:
            return True, float(e)
    return False, None


def test_abstract_torus2_exact():
    C = make_torus_grid_complex(2, (3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 2, 1]
    assert cohomology_at(C, eps) == [1, 2, 1]
    bc = persistent_homology(C)
    assert bc.betti_at(eps) == [1, 2, 1]
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(3)]
    assert ess == [1, 2, 1]


def test_abstract_torus3_exact():
    C = make_torus_grid_complex(3, (3, 3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 3, 3, 1]
    assert cohomology_at(C, eps) == [1, 3, 3, 1]
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(4)]
    assert ess == [1, 3, 3, 1]


def test_point_cloud_circle_recovers_h1():
    pts = G.circle_grid(24)
    ok, e = _reaches_beta1(pts, max_dim=2, target_beta1=1)
    assert ok, "circle: no epsilon with beta0=1 and beta1=1"


def test_point_cloud_torus2_recovers_two_loops():
    pts = G.product_torus_grid(2, 8)
    ok, e = _reaches_beta1(pts, max_dim=2, target_beta1=2)
    assert ok, "T^2: no epsilon with beta0=1 and beta1=2"


def test_point_cloud_torus3_recovers_three_loops():
    pts = G.product_torus_grid(3, 4)
    ok, e = _reaches_beta1(pts, max_dim=3, target_beta1=3)
    assert ok, "T^3: no epsilon with beta0=1 and beta1=3"


def test_rips_circle_betti_full():
    pts = G.circle_grid(24)
    D = pairwise_distances(pts)
    nn = _nearest_neighbor_mean(D)
    C = build_rips(pts, D, 1.2 * nn, max_dim=2)
    b = betti_at(C, 1.2 * nn)
    assert b[0] == 1
    assert b[1] == 1
