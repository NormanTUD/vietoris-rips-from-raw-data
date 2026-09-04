from __future__ import annotations

import numpy as np

from vrtda.beartype_guard import beartype_module
from vrtda.complexes import FilteredComplex, make_torus_grid_complex
from vrtda.cocycles import loops_1skeleton, persistent_loops
from vrtda.persistence import persistent_homology


def make_complex(simplices: list[tuple[int, ...]], values: list[float], kind: str = "test") -> FilteredComplex:
    # order by dim (faces always precede their simplices)
    order = sorted(range(len(simplices)), key=lambda i: (len(simplices[i]) - 1, i))
    simplexes = [simplices[i] for i in order]
    vals = [values[i] for i in order]
    dims = [len(s) - 1 for s in simplexes]
    return FilteredComplex(simplexes, np.array(vals), np.array(dims), kind, {})


def triangle_complex() -> FilteredComplex:
    s = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)]
    v = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3]
    return make_complex(s, v)


def test_triangle_single_loop() -> None:
    fc = triangle_complex()
    loops = loops_1skeleton(fc)
    assert len(loops) == 1
    assert loops[0].tokens == frozenset({0, 1, 2})
    assert len(loops[0].edges) == 3


def test_triangle_persistent_loop() -> None:
    fc = triangle_complex()
    bc = persistent_homology(fc)
    assert sum(1 for iv in bc.of_dim(1) if iv.is_essential) == 1
    loops = persistent_loops(fc, bc, essential_only=True)
    assert len(loops) == 1
    assert loops[0].tokens == frozenset({0, 1, 2})
    assert loops[0].death == float("inf")


def test_torus_two_essential_loops() -> None:
    fc = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(fc)
    assert sum(1 for iv in bc.of_dim(1) if iv.is_essential) == 2
    loops = persistent_loops(fc, bc, essential_only=True)
    assert len(loops) == 2
    assert len({lp.birth_simplex for lp in loops}) == 2
    for lp in loops:
        assert len(lp.tokens) >= 2


def test_two_disjoint_circles_two_loops() -> None:
    s = [(0,), (1,), (2,), (3,), (4,), (5,), (0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    v = [0.0] * 6 + [0.1] * 6
    fc = make_complex(s, v)
    loops = loops_1skeleton(fc)
    assert len(loops) == 2
    token_sets = sorted(lp.tokens for lp in loops)
    assert token_sets == [frozenset({0, 1, 2}), frozenset({3, 4, 5})]


def test_filled_triangle_no_essential_loop() -> None:
    s = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
    v = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5]
    fc = make_complex(s, v)
    bc = persistent_homology(fc)
    assert sum(1 for iv in bc.of_dim(1) if iv.is_essential) == 0
    assert persistent_loops(fc, bc, essential_only=True) == []
    assert len(loops_1skeleton(fc)) == 1  # basis still lists the (bound) cycle


beartype_module(__name__)
