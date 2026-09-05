from __future__ import annotations

import numpy as np
import pytest

from vrtda import FilteredComplex, persistent_homology
from vrtda.beartype_guard import beartype_module


def two_points_edge() -> FilteredComplex:
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (2.0, 1, (0, 1)),
    ])


def triangle_cycle() -> FilteredComplex:
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (2, 0)),
    ])


def test_two_points_edge_barcodes() -> None:
    bc = persistent_homology(two_points_edge())
    h0 = sorted((iv.birth, iv.death) for iv in bc.of_dim(0))
    # one component born at 0 infinite, one born at 0 killed at 2
    assert h0 == [(0.0, 2.0), (0.0, np.inf)]
    assert bc.of_dim(1) == []


def test_triangle_cycle_barcodes() -> None:
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


def test_interval_alive_at() -> None:
    bc = persistent_homology(triangle_cycle())
    loop = bc.of_dim(1)[0]
    assert loop.alive_at(1.0)
    assert loop.alive_at(100.0)
    assert not loop.alive_at(0.5)


def test_betti_at_from_barcode() -> None:
    bc = persistent_homology(triangle_cycle())
    assert bc.betti_at(0.0) == [3, 0]  # 3 isolated points
    assert bc.betti_at(1.0) == [1, 1]  # connected + one loop
    assert bc.betti_at(100.0) == [1, 1]


def test_interval_lengths_nonneg() -> None:
    bc = persistent_homology(triangle_cycle())
    for iv in bc.intervals:
        assert iv.death >= iv.birth - 1e-12


def test_torus_grid_essential_counts() -> None:
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(3)]
    assert ess == [1, 2, 1]


def test_3torus_essential_counts() -> None:
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(3, (3, 3, 3))
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(4)]
    assert ess == [1, 3, 3, 1]


def test_persistence_is_deterministic() -> None:
    from vrtda import pairwise_distances, build_rips
    from vrtda.generators import circle_grid
    pts = circle_grid(30)
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 0.5, max_dim=2)
    b1 = persistent_homology(C)
    b2 = persistent_homology(C)
    assert [iv.as_tuple() for iv in b1.intervals] == [iv.as_tuple() for iv in b2.intervals]


def test_betti_function_matches_betti_at() -> None:
    # The vectorised betti_function (birth <= eps < death, over the barcode) must agree
    # with the per-epsilon betti_at on a range of epsilons, including beyond eps_max.
    from vrtda.complexes import make_torus_grid_complex
    for C in (triangle_cycle(), make_torus_grid_complex(2, (3, 3))):
        bc = persistent_homology(C)
        emax = float(C.values.max())
        grid = [emax * f for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)]
        table = bc.betti_function(grid)
        for r, e in enumerate(grid):
            assert list(table[r]) == bc.betti_at(e), f"betti_function != betti_at at eps={e}"


def test_persistent_homology_progress_cb() -> None:
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    calls: list[tuple[int, int]] = []
    bc = persistent_homology(C, progress_cb=lambda j, n: calls.append((j, n)))
    assert calls, "progress callback never fired"
    assert calls[-1] == (C.n_simplices - 1, C.n_simplices)
    assert bc.max_dim() >= 2


beartype_module(__name__)
