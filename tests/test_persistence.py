import numpy as np
import pytest

from vrtda import FilteredComplex, persistent_homology


def two_points_edge():
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (2.0, 1, (0, 1)),
    ])


def triangle_cycle():
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (2, 0)),
    ])


def test_two_points_edge_barcodes():
    bc = persistent_homology(two_points_edge())
    h0 = sorted((iv.birth, iv.death) for iv in bc.of_dim(0))
    # one component born at 0 infinite, one born at 0 killed at 2
    assert h0 == [(0.0, 2.0), (0.0, np.inf)]
    assert bc.of_dim(1) == []


def test_triangle_cycle_barcodes():
    bc = persistent_homology(triangle_cycle())
    h0 = bc.of_dim(0)
    h1 = bc.of_dim(1)
    # H0: 1 infinite + 2 finite [0,1)
    ess0 = [iv for iv in h0 if iv.is_essential]
    fin0 = [iv for iv in h0 if not iv.is_essential]
    assert len(ess0) == 1
    assert len(fin0) == 2
    assert all(iv.death == 1.0 for iv in fin0)
    # H1: exactly one essential loop born at 1
    assert len(h1) == 1
    assert h1[0].is_essential
    assert h1[0].birth == 1.0


def test_interval_alive_at():
    bc = persistent_homology(triangle_cycle())
    loop = bc.of_dim(1)[0]
    assert loop.alive_at(1.0)
    assert loop.alive_at(100.0)
    assert not loop.alive_at(0.5)


def test_betti_at_from_barcode():
    bc = persistent_homology(triangle_cycle())
    assert bc.betti_at(0.0) == [3, 0]  # 3 isolated points
    assert bc.betti_at(1.0) == [1, 1]  # connected + one loop
    assert bc.betti_at(100.0) == [1, 1]


def test_interval_lengths_nonneg():
    bc = persistent_homology(triangle_cycle())
    for iv in bc.intervals:
        assert iv.death >= iv.birth - 1e-12


def test_torus_grid_essential_counts():
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(3)]
    assert ess == [1, 2, 1]


def test_3torus_essential_counts():
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(3, (3, 3, 3))
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(4)]
    assert ess == [1, 3, 3, 1]


def test_persistence_is_deterministic():
    from vrtda import pairwise_distances, build_rips
    from vrtda.generators import circle_grid
    pts = circle_grid(30)
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 0.5, max_dim=2)
    b1 = persistent_homology(C)
    b2 = persistent_homology(C)
    assert [iv.as_tuple() for iv in b1.intervals] == [iv.as_tuple() for iv in b2.intervals]
