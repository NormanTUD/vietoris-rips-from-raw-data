from __future__ import annotations

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
from vrtda.beartype_guard import beartype_module
from vrtda.complexes import make_torus_grid_complex


def _nearest_neighbor_mean(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def _reaches_beta1(pts: np.ndarray, max_dim: int, target_beta1: int, hi: float = 2.0) -> bool:
    """Does some epsilon in [0.9*nn, hi*nn] give beta_0=1 and beta_1 == target?"""
    D = pairwise_distances(pts)
    nn = _nearest_neighbor_mean(D)
    C = build_rips(pts, D, hi * nn, max_dim=max_dim)
    bc = persistent_homology(C)
    epsilons = np.linspace(0.9 * nn, hi * nn, 40)
    for e in epsilons:
        b = bc.betti_at(e)
        if b[0] == 1 and len(b) > 1 and b[1] == target_beta1:
            return True
    return False


def test_abstract_torus2_exact() -> None:
    C = make_torus_grid_complex(2, (3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 2, 1]
    assert cohomology_at(C, eps) == [1, 2, 1]
    bc = persistent_homology(C)
    assert bc.betti_at(eps) == [1, 2, 1]
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(3)]
    assert ess == [1, 2, 1]


def test_abstract_torus3_exact() -> None:
    C = make_torus_grid_complex(3, (3, 3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 3, 3, 1]
    assert cohomology_at(C, eps) == [1, 3, 3, 1]
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(4)]
    assert ess == [1, 3, 3, 1]


def test_point_cloud_circle_recovers_h1() -> None:
    assert _reaches_beta1(G.circle_grid(24), max_dim=2, target_beta1=1, hi=2.0)


def test_point_cloud_torus2_recovers_two_loops() -> None:
    assert _reaches_beta1(G.product_torus_grid(2, 8), max_dim=2, target_beta1=2, hi=2.0)


def test_point_cloud_torus3_recovers_three_loops() -> None:
    assert _reaches_beta1(G.product_torus_grid(3, 5), max_dim=3, target_beta1=3, hi=1.6)


def test_rips_circle_betti_full() -> None:
    pts = G.circle_grid(24)
    D = pairwise_distances(pts)
    nn = _nearest_neighbor_mean(D)
    C = build_rips(pts, D, 1.2 * nn, max_dim=2)
    b = betti_at(C, 1.2 * nn)
    assert b[0] == 1
    assert b[1] == 1


beartype_module(__name__)
